import os
import io
import csv
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import fingerprint as fp

st.set_page_config(page_title="EE200 Audio Fingerprinting", layout="wide")

SONGS_DIR = "songs"
DB_PATH = "db.pkl"

@st.cache_resource
def get_db():
    if os.path.exists(DB_PATH):
        return fp.FingerprintDB.load(DB_PATH)
    db = fp.FingerprintDB()
    if os.path.isdir(SONGS_DIR):
        n = db.build_from_folder(SONGS_DIR)
        if n > 0:
            db.save(DB_PATH)
    return db

db = get_db()

def plot_spectrogram(samples):
    S, S_db, freqs, times = fp.compute_spectrogram(samples)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.pcolormesh(times, freqs, S_db, shading="gouraud", cmap="magma")
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.tight_layout()
    return fig, S

def plot_constellation(S):
    peaks = fp.find_peaks(S)
    fig, ax = plt.subplots(figsize=(8, 3))
    if peaks:
        t = [p[0] for p in peaks]; f = [p[1] for p in peaks]
        ax.scatter(t, f, s=4, c="#2dd4bf")
    ax.set_xlabel("time (frame)"); ax.set_ylabel("freq (bin)")
    ax.set_title(f"Constellation - {len(peaks)} peaks")
    fig.tight_layout()
    return fig, peaks

def plot_offset_hist(details):
    fig, ax = plt.subplots(figsize=(8, 3))
    hist = details.get("offset_hist", {})
    if hist:
        offs = list(hist.keys()); cnts = list(hist.values())
        ax.bar(offs, cnts, color="#2dd4bf")
        ax.axvline(details["best_offset"], color="red", ls="--",
                   label=f"peak offset = {details['best_offset']}")
        ax.legend()
    ax.set_xlabel("time offset (db_anchor - query_anchor)")
    ax.set_ylabel("matching hashes")
    ax.set_title("Offset histogram - a true match spikes at one offset")
    fig.tight_layout()
    return fig

def identify_samples(samples):
    qhashes, qpeaks, S = fp.fingerprint_samples(samples)
    best, score, scores, details = fp.match(qhashes, db, return_details=True)
    return best, score, details, S, qpeaks

st.title("EE200: Audio Fingerprinting")
st.caption("Signals, Systems & Networks - Project. "
           "Index songs as spectrogram fingerprints, then identify any short clip.")

tab_lib, tab_id, tab_batch = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])

with tab_lib:
    st.subheader("Indexed library")
    if not db.songs:
        st.warning(
            f"No songs indexed. Put the provided .mp3 files in a `{SONGS_DIR}/` "
            "folder next to this app and restart, or ship a prebuilt `db.pkl`."
        )
    else:
        st.write(f"**{len(db.songs)} songs** indexed, "
                 f"**{sum(db.songs.values()):,} total hashes**.")
        cols = st.columns(3)
        for i, (label, n) in enumerate(sorted(db.songs.items())):
            with cols[i % 3]:
                st.markdown(f"**{label}**  \n{n:,} hashes")

with tab_id:
    st.subheader("Identify a clip")
    up = st.file_uploader("Upload a query clip",
                          type=["wav", "mp3", "flac", "ogg", "m4a"], key="single")
    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=os.path.splitext(up.name)[1]) as tmp:
            tmp.write(up.read()); tmp_path = tmp.name
        samples = fp.load_audio(tmp_path)
        os.unlink(tmp_path)

        best, score, details, S, qpeaks = identify_samples(samples)

        if best is None:
            st.error("No match found (empty database or no overlapping hashes).")
        else:
            st.success(f"MATCH FOUND: **{best}**  (cluster score {score})")
            runner = details["ranked"][1] if len(details["ranked"]) > 1 else None
            if runner:
                st.caption(f"runner-up: {runner[0]} (score {runner[1]})")

        st.markdown("**Step 1 - Spectrogram**")
        fig, S2 = plot_spectrogram(samples); st.pyplot(fig)

        st.markdown("**Step 2 - Constellation of peaks**")
        figc, _ = plot_constellation(S2); st.pyplot(figc)

        if best is not None:
            st.markdown("**Step 3 - Offset histogram (the deciding vote)**")
            st.pyplot(plot_offset_hist(details))

            st.markdown("**Candidate scores**")
            for label, sc in details["ranked"][:6]:
                st.write(f"{label}: {sc}")

with tab_batch:
    st.subheader("Batch mode")
    st.caption("Upload several clips; download results.csv with columns "
               "`filename,prediction`.")
    ups = st.file_uploader("Upload query clips",
                           type=["wav", "mp3", "flac", "ogg", "m4a"],
                           accept_multiple_files=True, key="batch")
    if ups and st.button("Run batch"):
        rows = []
        prog = st.progress(0.0)
        for i, f in enumerate(ups):
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=os.path.splitext(f.name)[1]) as tmp:
                tmp.write(f.read()); tmp_path = tmp.name
            samples = fp.load_audio(tmp_path); os.unlink(tmp_path)
            qhashes, _, _ = fp.fingerprint_samples(samples)
            best, _, _ = fp.match(qhashes, db)
            rows.append((f.name, best if best else ""))
            prog.progress((i + 1) / len(ups))

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["filename", "prediction"])
        w.writerows(rows)
        st.dataframe(rows)
        st.download_button("Download results.csv", buf.getvalue(),
                           file_name="results.csv", mime="text/csv")

