# StarFire

### Explore NASA‚Äôs microgravity fire-safety research through your own questions.

StarFire is a working research-discovery prototype that helps users find potentially relevant NASA Physical Sciences Informatics (PSI) investigations, understand their objectives, and decide what evidence to inspect next.

Ask about material flammability, smoke detection, or fire suppression. StarFire returns a ranked shortlist with question-specific AI notes, original investigation metadata, and source links.

> Research discovery‚Äînot a safety verdict. Ranked matches and generated interpretations are not verified scientific conclusions or confirmation of dataset suitability.

## Why StarFire?

Finding an investigation is only the first step. Researchers also need to understand why it may relate to their question, distinguish intended work from reported findings, and identify a useful next action.

StarFire brings those steps into one interface while keeping the underlying metadata accessible for checking.

## Features

- **Natural-language search:** Describe a research question and retrieve ranked investigation candidates.
- **Question-specific research notes:** Automatically generate a concise note for each retrieved investigation:
  - **Connection:** How the investigation relates to the question, including indirect or partial matches.
  - **Useful evidence:** Relevant documented methods, conditions, hypotheses, or reported findings.
  - **Interpretation:** A cautious explanation of what the evidence could mean for the research question.
  - **Next step:** One concrete research action tied to the investigation and question.
- **Investigation evidence:** Inspect original metadata fields and follow available source links.
- **Shortlist comparison:** Compare investigation titles, research areas, hardware, and source links where available.
- **Research exports:** Download the shortlist as CSV or a readable TXT research brief.
- **Local text generation:** Use a downloaded language model without a hosted inference API for the research notes.
- **Session reuse:** Reuse generated notes for the same question, metadata, model, and prompt version within the current app session.

## Example questions

- How does airflow affect flame spread over solid materials in microgravity?
- Which investigations could help me study smoke detection?
- What research examines oxygen concentration and material flammability?
- Which investigations are relevant to fire suppression in microgravity?

These are example searches, not guarantees that the catalog contains evidence sufficient to answer them.

## How it works

1. **Ask:** Enter a research question and choose the number of investigations to retrieve.
2. **Retrieve:** The project‚Äôs `PSIInvestigationRecommender` ranks investigations from `investigations.csv`.
3. **Generate:** A local language model receives the question and selected investigation metadata excerpts.
4. **Inspect:** Read the generated note and open the original metadata and source link to check its claims.
5. **Export:** Save a shortlist or research brief for follow-up work.

The AI prompt instructs the model to use only supplied metadata, distinguish hypotheses and objectives from findings, and avoid unsupported claims. These instructions reduce neither the need for source checking nor the possibility of errors.

## Technology

| Component | Implementation |
| --- | --- |
| Interface | Streamlit |
| Data handling and CSV export | pandas |
| Investigation ranking | Project module: `PSIInvestigationRecommender` |
| Local inference | Hugging Face Transformers and PyTorch |
| Language model | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Inference device | Apple MPS when available; otherwise CPU |
| Investigation catalog | Local `investigations.csv` |

## Run locally

### Prerequisites

- Python and a virtual environment suitable for the project‚Äôs dependencies.
- The repository‚Äôs investigation catalog and recommender module.
- Sufficient disk space and memory for the local language model.
- Internet access for initial dependency and model downloads.

### 1. Clone the repository

```bash
git clone https://github.com/soukainaRhazzafe/microgravity-fire-dashboard.git
cd microgravity-fire-dashboard
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the repository includes a maintained `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The interface and local generator use `streamlit`, `pandas`, `torch`, `transformers`, and `safetensors`. The recommender may require additional packages; its imports and the project dependency file determine the complete environment.

### 4. Download the local model

The application expects the tokenizer and model files under:

```text
models/smollm2/
```

If using the project‚Äôs model-download script, run:

```bash
python download_model.py
```

The local generator loads the files from that directory with `local_files_only=True`; it does not download missing model files automatically during generation.

### 5. Start the app

```bash
streamlit run app.py
```

Open the local address printed in the terminal. Press `Ctrl+C` in the terminal to stop the app.

## Application files

The app expects these files or directories relative to `app.py`:

```text
app.py                    # Streamlit interface and local text-generation workflow
recommender.py            # Defines PSIInvestigationRecommender
investigations.csv        # Investigation catalog
models/
‚îî‚îÄ‚îÄ smollm2/               # Downloaded model and tokenizer files
```

A model-download script and dependency file support setup when included in the repository. Large model files do not need to be committed to Git to run the app locally.

## Data and attribution

The application uses investigation metadata attributed to NASA PSI, with links to original investigation pages where available. Fields may include objectives, hypotheses, experimental approaches, reported results, hardware, and research context.

- Source records remain distinct from AI-generated interpretation.
- Some metadata supplied to the model is excerpted to fit its context window.
- A source link does not mean every generated sentence has been verified against the linked page.
- CSV and TXT exports are research-discovery artifacts, not experimental training datasets.

StarFire is an independent prototype. It is not affiliated with or endorsed by NASA. Source materials and model weights remain subject to their respective terms; their inclusion does not establish a license for this repository.

## Limitations

- Retrieval quality has not been established by a formal evaluation in this prototype.
- Ranking identifies potentially relevant records; it does not establish scientific relevance with certainty.
- The local model can produce inaccurate, incomplete, repetitive, or unsupported text despite its grounding instructions.
- Metadata may be incomplete. Absence from an excerpt is not evidence that a full investigation lacks a measurement or result.
- A variable mentioned in metadata is not necessarily an available data column.
- Relevance does not establish that experimental files contain linked inputs and outcomes suitable for modeling.
- More requested investigations require more sequential generation work. Performance depends on hardware and response length.
- The interface does not provide operational fire-safety advice, validated safety scores, or verified predictions.
- Local note generation does not imply that the recommender and all setup steps work fully offline; that depends on their dependencies and cached resources.

## Next steps

- Evaluate retrieval on a documented set of research questions.
- Assess generated claims against their supporting metadata.
- Improve field-level traceability between generated interpretations and source evidence.
- Inspect experimental files before making claims about modeling suitability.
- Gather user feedback on the shortlist and research-brief workflow.

## Project status

Working hackathon prototype. The focus is a clear, inspectable journey from a research question to relevant investigation candidates and a concrete next research step.

