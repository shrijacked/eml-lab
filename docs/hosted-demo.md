# Hosted Demo

EML Lab is local-first, but the Streamlit dashboard is now packaged for hosted demos.
The hosted app should run the same package APIs as the CLI; no math logic is duplicated
for deployment.

## Streamlit Community Cloud

Use these settings:

- Repository: this repo
- Branch: `main`
- Main file path: `src/eml_lab/app.py`
- Python version: `runtime.txt`
- Dependencies: `requirements.txt`

The app is CPU-first. PySR remains optional; if Julia or PySR is missing, comparison
screens show install guidance instead of failing the app.

## Docker

Build the image:

```bash
docker build -t eml-lab .
```

Run the dashboard:

```bash
docker run --rm -p 8501:8501 eml-lab
```

Then open:

```text
http://localhost:8501
```

The container uses writable temp directories for Matplotlib and cache files so hosted
and sandboxed environments do not fail when generating plots. It also preinstalls
Torch from the PyTorch CPU wheel index before installing EML Lab, keeping the demo
image aligned with the repo's CPU-first scope.

## Local Smoke

The non-container smoke path is:

```bash
python -m pip install -e .
python -m eml_lab app --dry-run
streamlit run src/eml_lab/app.py --server.headless true --server.port 8501
```
