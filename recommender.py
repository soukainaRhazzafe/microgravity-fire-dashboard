import threading

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer


class PSIInvestigationRecommender:
    """
    Two-stage investigation search:

    1. Retrieve candidate investigations using embedding similarity.
    2. Rerank candidate passages against the question jointly.

    All research areas remain searchable.
    Scores indicate relative relevance, not dataset suitability.
    """

    SEARCH_FIELDS = [
        "objective",
        "approach",
        "results",
        "hypothesis",
        "research_impacts",
    ]

    def __init__(
        self,
        csv_path,
        candidate_pool=20,
        passages_per_investigation=3,
    ):
        self.candidate_pool = max(1, int(candidate_pool))
        self.passages_per_investigation = max(
            1,
            int(passages_per_investigation),
        )

        self.lock = threading.Lock()

        self.df = pd.read_csv(csv_path).fillna("")

        if "title" not in self.df.columns:
            raise ValueError("The CSV must contain a 'title' column.")

        columns = [
            "title",
            "research_area",
            "hardware",
            "source_url",
            *self.SEARCH_FIELDS,
        ]

        for column in columns:
            if column not in self.df.columns:
                self.df[column] = ""

            self.df[column] = (
                self.df[column]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "nan": "",
                        "None": "",
                        "<NA>": "",
                    }
                )
            )

        self.df["research_area"] = (
            self.df["research_area"]
            .str.replace(r"^\s*:\s*", "", regex=True)
            .str.strip()
        )

        # Deduplicate only records with the same nonempty source URL.
        repeated_url = (
            self.df["source_url"].ne("")
            & self.df["source_url"].duplicated()
        )

        self.df = (
            self.df.loc[~repeated_url]
            .reset_index(drop=True)
        )

        if self.df.empty:
            raise ValueError("The CSV contains no investigations.")

        self.retriever = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            max_length=512,
        )

        self.passages = []
        self.passage_owners = []
        self.passage_fields = []

        for row_number, row in self.df.iterrows():
            title = row["title"]
            area = row["research_area"]

            # Keep the title itself available as a searchable passage.
            if title:
                self._add_passage(
                    f"Title: {title}\nResearch area: {area}",
                    row_number,
                    "title",
                )

            for field in self.SEARCH_FIELDS:
                if not row[field]:
                    continue

                label = field.replace("_", " ").capitalize()

                prefix = (
                    f"Title: {title}\n"
                    f"Research area: {area}\n"
                    f"{label}: "
                )

                for passage in self._make_passages(
                    prefix,
                    row[field],
                ):
                    self._add_passage(
                        passage,
                        row_number,
                        field,
                    )

        if not self.passages:
            raise ValueError("No searchable investigation text was found.")

        self.passage_owners = np.asarray(
            self.passage_owners,
            dtype=int,
        )

        self.embeddings = self.retriever.encode(
            self.passages,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        print(
            f"Indexed {len(self.df)} investigations "
            f"across {len(self.passages)} passages."
        )

    def _add_passage(self, text, owner, field):
        self.passages.append(text)
        self.passage_owners.append(owner)
        self.passage_fields.append(field)

    def _make_passages(self, prefix, text):
        """
        Split field text into token-sized passages.

        Reserve room for the title/field prefix in each passage.
        """
        tokenizer = self.retriever.tokenizer

        max_length = int(self.retriever.max_seq_length)
        special_tokens = tokenizer.num_special_tokens_to_add(
            pair=False
        )

        # Bound unusually long titles so some room remains for content.
        prefix_ids = tokenizer.encode(
            prefix,
            add_special_tokens=False,
        )[:70]

        prefix_text = tokenizer.decode(
            prefix_ids,
            skip_special_tokens=True,
        )

        # Extra margin accounts for tokenization at the text boundary.
        budget = max_length - len(prefix_ids) - special_tokens - 12
        budget = max(20, min(170, budget))

        text_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        overlap = min(25, budget // 4)
        step = budget - overlap

        for start in range(0, len(text_ids), step):
            chunk_ids = text_ids[start:start + budget]

            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
            )

            yield f"{prefix_text}\n{chunk_text}"

            if start + budget >= len(text_ids):
                break

    def recommend(self, task, top_k=5):
        task = str(task).strip()
        top_k = int(top_k)

        if not task:
            raise ValueError("Please enter a research question.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Avoid silently truncating a long research question.
        query_length = len(
            self.retriever.tokenizer.encode(
                task,
                add_special_tokens=True,
            )
        )

        if query_length > self.retriever.max_seq_length:
            raise ValueError(
                "Please shorten the research question to a few sentences."
            )

        with self.lock:
            return self._recommend(task, top_k)

    def _recommend(self, task, top_k):
        query_embedding = self.retriever.encode(
            [task],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        # Normalized dot products are cosine similarities.
        passage_scores = self.embeddings @ query_embedding

        # First-stage investigation score: its strongest matching passage.
        investigation_scores = np.full(
            len(self.df),
            -np.inf,
            dtype=float,
        )

        np.maximum.at(
            investigation_scores,
            self.passage_owners,
            passage_scores,
        )

        candidate_count = min(
            len(self.df),
            max(self.candidate_pool, top_k),
        )

        candidate_ids = np.argsort(
            -investigation_scores,
            kind="stable",
        )[:candidate_count]

        # Send only a few promising passages per investigation
        # to the more expensive cross-encoder.
        selected_passage_ids = []

        for investigation_id in candidate_ids:
            indices = np.flatnonzero(
                self.passage_owners == investigation_id
            )

            ordered_indices = indices[
                np.argsort(
                    -passage_scores[indices],
                    kind="stable",
                )
            ]

            selected_passage_ids.extend(
                ordered_indices[
                    :self.passages_per_investigation
                ].tolist()
            )

        pairs = [
            [task, self.passages[index]]
            for index in selected_passage_ids
        ]

        rerank_scores = np.asarray(
            self.reranker.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            )
        ).reshape(-1)

        # Aggregate passage relevance back to investigation relevance.
        best_passage = {}

        for passage_id, score in zip(
            selected_passage_ids,
            rerank_scores,
        ):
            owner = int(self.passage_owners[passage_id])
            score = float(score)

            if (
                owner not in best_passage
                or score > best_passage[owner]["score"]
            ):
                best_passage[owner] = {
                    "score": score,
                    "passage_id": passage_id,
                }

        result = self.df.iloc[candidate_ids].copy()

        result["retrieval_score"] = [
            float(investigation_scores[index])
            for index in candidate_ids
        ]

        # Keep "score" for compatibility with app.py.
        result["score"] = [
            best_passage[int(index)]["score"]
            for index in candidate_ids
        ]

        result["matched_field"] = [
            self.passage_fields[
                best_passage[int(index)]["passage_id"]
            ]
            for index in candidate_ids
        ]

        result["matched_passage"] = [
            self.passages[
                best_passage[int(index)]["passage_id"]
            ]
            for index in candidate_ids
        ]

        return (
            result.sort_values(
                ["score", "retrieval_score"],
                ascending=[False, False],
                kind="mergesort",
            )
            .head(top_k)
            .reset_index(drop=True)
        )