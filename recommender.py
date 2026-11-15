import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class PSIInvestigationRecommender:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.df["research_area"] = (self.df["research_area"].astype(str).str.replace(r"^\s*:\s*","",regex=True).str.strip()
        )

        self.df["search_text"] = (
            self.df["title"].fillna("")
            + " "
            + self.df["approach"].fillna("")
            + " "
            + self.df["objective"].fillna("")
            + " "
            + self.df["hypothesis"].fillna("")
            + " "
            + self.df["research_impacts"].fillna("")
            + " "
            + self.df["research_area"].fillna("")
        )

        self.embeddings = self.model.encode(
            self.df["search_text"].tolist(),
            show_progress_bar=False
        )

    def recommend(
        self,
        task,
        top_k=5
    ):

        query_embedding = self.model.encode(
            [task]
        )

        scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        result = self.df.copy()

        result["score"] = scores

        result = result.sort_values(
            "score",
            ascending=False
        )

        return result.head(top_k)