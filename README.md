# StarFire

Explore NASA's microgravity fire-safety research through your own questions.

I built StarFire to make it easier to find relevant investigations and figure out what to look at next. Ask about flame spread, material flammability, smoke detection, or fire suppression. The app returns a ranked shortlist with original metadata, source links, and AI-generated notes focused on your question.

## What it does

- Finds potentially relevant investigations from a local NASA PSI catalog.
- Explains the connection to your question, useful evidence, and possible interpretations.
- Suggests a concrete next research step for each investigation.
- Keeps original metadata and source links available for checking.
- Lets you compare investigations and export a CSV shortlist or TXT research brief.
- Generates notes locally with SmolLM2, using Transformers and PyTorch.

## Run it locally

Clone the repository:

```bash
git clone https://github.com/soukainaRhazzafe/microgravity-fire-dashboard.git
cd microgravity-fire-dashboard
```

Create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the local model:

```bash
python download_model.py
```

The app expects the model and tokenizer files in `models/smollm2/` and the investigation catalog at `investigations.csv`, next to `app.py`.

Start the app:

```bash
streamlit run app.py
```

Open the local address shown in the terminal. Press `Ctrl+C` to stop it.

Start with three investigations per search. Notes are generated one at a time and reused within the session when the question and metadata have not changed.

## Main files

- `app.py`: interface, research notes, and exports.
- `recommender.py`: investigation retrieval and ranking.
- `investigations.csv`: catalog used by the app.
- `scraper.py`: collects investigation records separately from the app.
- `download_model.py`: downloads the local language model.
- `models/smollm2/`: downloaded model files.

## Scraper setup

Only needed if you want to collect the metadata again:

```bash
python -m playwright install chromium
python scraper.py
```

Check the scraper's output location before replacing the catalog used by the app. Running the scraper is not required to launch the app with an existing catalog.

## What to keep in mind

This is a hackathon prototype, not a validated research tool. AI notes can be wrong. Check them against the metadata and original sources.

A relevant investigation is not automatically a usable modeling dataset. Mentioned variables are not confirmed data columns, and hypotheses are not findings. The exported brief is a research aid, not experimental data or fire-safety advice.

The next steps are to evaluate retrieval quality, check generated claims against their sources, and inspect experimental files before making modeling claims.

## Sources

Investigation metadata is attributed to NASA Physical Sciences Informatics (PSI), with original links included where available. The local language model is `HuggingFaceTB/SmolLM2-1.7B-Instruct`.

StarFire is an independent project, not affiliated with or endorsed by NASA. Source materials and model weights remain subject to their respective terms.
