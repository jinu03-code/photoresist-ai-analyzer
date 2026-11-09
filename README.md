# Photoresist Exposure-Time Analyzer

Streamlit app for photoresist microscope-image analysis.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Deploy this repository with Streamlit Community Cloud. Entrypoint: `app.py`.

## Important storage note
Uploaded and analyzed images are written to `pr_saved_data/` on the running machine.
On hosted Streamlit Community Cloud, local runtime storage should not be treated as
permanent storage. For reliable long-term image retention across restarts/redeploys,
connect a persistent external storage service before using this app for important data.
Do not upload confidential or unpublished research images to a public demo without
reviewing access and data-retention requirements.
