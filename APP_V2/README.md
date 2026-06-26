# EE200 Q3 - Audio Fingerprinting

Shazam-style audio identifier (Q3A engine) wrapped in a Streamlit app (Q3B).

## Files
- `fingerprint.py` - the engine: spectrogram -> constellation -> paired hashes -> offset-histogram match.
- `app.py` - Streamlit app with LIBRARY / IDENTIFY / BATCH tabs.
- `build_db.py` - indexes the song folder once into `db.pkl`.
- `requirements.txt` - dependencies.

## Setup (local)
1. Download the provided songs from the course Drive link.
2. Put all `.mp3` files in a folder named `songs/` next to these files.
   IMPORTANT: do NOT rename them - the filename without extension is the label.
3. Install deps:  `pip install -r requirements.txt`
4. Build the database once:  `python build_db.py`
5. Run the app:  `streamlit run app.py`

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo, INCLUDING `songs/` and `db.pkl`
   (so the app works immediately without re-indexing).
   - If the audio folder is too large for the repo, commit only `db.pkl`
     (the app loads it directly; songs are not needed at query time).
2. On https://share.streamlit.io connect the repo, set `app.py` as the entrypoint, deploy.
3. Submit the live URL and the source-code link in your PDF.

## Modes (per the brief)
- Single-clip: shows spectrogram, constellation, offset histogram, and the match.
- Batch: produces `results.csv` with columns `filename,prediction` (prediction = song filename without extension).
