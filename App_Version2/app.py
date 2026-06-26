import os
import io
import csv
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
import fingerprint as fp
import pickle

st.set_page_config(page_title="Audio Fingerprinting", layout="wide")
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
@st.cache_resource
def get_single_peak_index():
    # Primary: Load the pre-computed file (Streamlit will use this)
    if os.path.exists("sp_index.pkl"):
        with open("sp_index.pkl", "rb") as f:
            return pickle.load(f)
            
    # Fallback: Just in case you run locally without building the pkl first
    if os.path.isdir(SONGS_DIR):
        return fp.build_single_peak_index_from_folder(SONGS_DIR)
    
    return {}

def plot_dft_magnitude(samples, sr=fp.SAMPLE_RATE):
    freqs, mag = fp.compute_dft_magnitude(samples, sr=sr)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(freqs, mag, color="#f97316", linewidth=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Full-song DFT Magnitude")
    ax.set_xlim(0, sr / 2)
    fig.tight_layout()
    return fig

def plot_spectrogram(samples):
    S, S_db, freqs, times = fp.compute_spectrogram(samples)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(
        S_db, aspect="auto", origin="lower", cmap="magma",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.tight_layout()
    return fig, S

def plot_constellation(peaks, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 3))
    if peaks:
        t = [p[0] for p in peaks]
        f = [p[1] for p in peaks]
        ax.scatter(t, f, s=4, c="#2dd4bf", alpha=0.8)
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel("Freq (bin)")
    ax.set_title(f"Constellation map – {len(peaks)} peaks{title_suffix}")
    fig.tight_layout()
    return fig

def plot_offset_hist(details):
    fig, ax = plt.subplots(figsize=(8, 3))
    hist = details.get("offset_hist", {})
    if hist:
        offs = list(hist.keys())
        cnts = list(hist.values())
        ax.bar(offs, cnts, color="#2dd4bf", width=1.0)
        ax.axvline(
            details["best_offset"], color="red", ls="--",
            label=f"peak offset = {details['best_offset']}",
        )
        ax.legend()
    ax.set_xlabel("Time offset")
    ax.set_ylabel("Matching hashes")
    ax.set_title("Offset histogram")
    fig.tight_layout()
    return fig

def plot_candidate_scores(ranked, top_n=8):
    top = ranked[:top_n]
    labels = [r[0] for r in top][::-1]
    scores = [r[1] for r in top][::-1]
    fig, ax = plt.subplots(figsize=(8, max(2, 0.45 * len(labels))))
    bars = ax.barh(labels, scores, color="#818cf8")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Cluster score")
    ax.set_title("Candidate scores")
    fig.tight_layout()
    return fig

def identify_samples(samples):
    qhashes, peaks, S = fp.fingerprint_samples(samples)
    best, score, _, details = fp.match(qhashes, db, return_details=True)
    return best, score, details, peaks, S

st.title("EE200: Audio Fingerprinting")
st.caption("Signals, Systems & Networks – Course Project")

tab_lib, tab_id, tab_batch = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])

with tab_lib:
    st.subheader("Indexed song library")
    if not db.songs:
        st.warning("No songs indexed. Add audio files to the `songs/` folder and restart, or provide a `db.pkl`.")
    else:
        total_hashes = sum(db.songs.values())
        st.write(f"**{len(db.songs)} songs** indexed · **{total_hashes:,} total hashes**")

        lib_df = pd.DataFrame(
            [(lbl, f"{n:,}") for lbl, n in sorted(db.songs.items())],
            columns=["Song", "Hashes"],
        )
        st.dataframe(lib_df, use_container_width=True, hide_index=True)

with tab_id:
    st.subheader("Identify a clip")
    up = st.file_uploader(
        "Upload a query clip",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="single",
    )
    if up is not None:
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up.read())
            tmp_path = tmp.name

        with st.spinner("Fingerprinting..."):
            samples = fp.load_audio(tmp_path)
        os.unlink(tmp_path)

        with st.spinner("Matching against database..."):
            best, score, details, peaks, S = identify_samples(samples)

        if best is None:
            st.error("No match found.")
        else:
            st.success(f"✅ MATCH FOUND: **{best}** (cluster score {score})")
            if len(details["ranked"]) > 1:
                runner = details["ranked"][1]
                st.caption(f"Runner-up: {runner[0]} (score {runner[1]})")

        st.divider()

        st.markdown("**Step 0 – Full-song DFT Magnitude**")
        with st.spinner("Computing DFT..."):
            fig_dft = plot_dft_magnitude(samples)
        st.pyplot(fig_dft)
        plt.close(fig_dft)

        st.markdown("**Step 1 – Spectrogram**")
        with st.spinner("Computing spectrogram..."):
            fig_spec, S2 = plot_spectrogram(samples)
        st.pyplot(fig_spec)
        plt.close(fig_spec)

        st.markdown("**Step 2 – Constellation of strongest peaks**")
        fig_const = plot_constellation(peaks)
        st.pyplot(fig_const)
        plt.close(fig_const)

        if best is not None:
            st.markdown("**Step 3 – Offset histogram**")
            fig_hist = plot_offset_hist(details)
            st.pyplot(fig_hist)
            plt.close(fig_hist)

            st.markdown("**Candidate scores (pair-hash matching)**")
            fig_scores = plot_candidate_scores(details["ranked"])
            st.pyplot(fig_scores)
            plt.close(fig_scores)

        st.divider()
        st.markdown("**Q3A Comparison – Single-peak vs pair-hash matching**")

        sp_index_approx = {}
        for h_str, entries in db.index.items():
            for label, t in entries:
                if label not in sp_index_approx:
                    sp_index_approx[label] = set()

        qfreqs = set(f for _, f in peaks)
        sp_scores = {}

        song_freq_sets = defaultdict(set)
        for h_str, entries in db.index.items():
            for label, t in entries:
                song_freq_sets[label].add(t % (fp.N_FFT // 2 + 1))

        st.divider()
        st.markdown(
            "**Q3A Comparison – Single-peak matching vs pair-hash matching**\n\n"
            "Single-peak matching ignores time alignment and pairing, so many songs "
            "share the same frequency bins by chance, yielding unreliable scores."
        )

        
        with st.spinner("Computing single-peak scores..."):
            sp_best_label, sp_best_score, sp_scores = fp.match_single_peaks(peaks, sp_index)
            
        sp_ranked = sorted(sp_scores.items(), key=lambda kv: kv[1], reverse=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*Pair-hash scores (top 6)*")
            ph_df = pd.DataFrame(details["ranked"][:6], columns=["Song", "Score"])
            st.dataframe(ph_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("*Single-peak scores (top 6)*")
            sp_df = pd.DataFrame(sp_ranked[:6], columns=["Song", "Score"])
            st.dataframe(sp_df, hide_index=True, use_container_width=True)

        if best is not None:
            sp_best = sp_ranked[0][0] if sp_ranked else "—"
            if sp_best == best:
                st.info(f"Both methods agree on **{best}**.")
            else:
                st.warning(f"Pair-hash says **{best}**, single-peak says **{sp_best}**.")

with tab_batch:
    st.subheader("Batch mode")
    ups = st.file_uploader(
        "Upload query clips",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        key="batch",
    )

    if ups and st.button("Run batch"):
        rows = []
        prog = st.progress(0.0)
        status_box = st.empty()

        for i, f in enumerate(ups):
            status_box.write(f"Processing `{f.name}`...")
            suffix = os.path.splitext(f.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            samples = fp.load_audio(tmp_path)
            os.unlink(tmp_path)

            qhashes, _, _ = fp.fingerprint_samples(samples)
            best, _, _ = fp.match(qhashes, db)
            rows.append({"filename": f.name, "prediction": best if best else ""})
            prog.progress((i + 1) / len(ups))

        status_box.empty()

        result_df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["filename", "prediction"])
        w.writeheader()
        w.writerows(rows)

        st.download_button(
            "⬇ Download results.csv",
            buf.getvalue(),
            file_name="results.csv",
            mime="text/csv",
        )