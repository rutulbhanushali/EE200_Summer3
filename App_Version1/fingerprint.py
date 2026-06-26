import os
import glob
import pickle
import hashlib
from collections import defaultdict

import numpy as np
import librosa

SAMPLE_RATE = 11025
N_FFT = 1024
HOP = 512
PEAK_NEIGH_TIME = 8
PEAK_NEIGH_FREQ = 8
MIN_AMP_PERCENTILE = 60
FAN_VALUE = 15
MIN_DT = 1
MAX_DT = 80

def compute_spectrogram(samples, sr=SAMPLE_RATE, n_fft=N_FFT, hop=HOP):
    S = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return S, S_db, freqs, times

def load_audio(path, sr=SAMPLE_RATE):
    samples, _ = librosa.load(path, sr=sr, mono=True)
    return samples

def find_peaks(S, neigh_t=PEAK_NEIGH_TIME, neigh_f=PEAK_NEIGH_FREQ,
               amp_pct=MIN_AMP_PERCENTILE):
    from scipy.ndimage import maximum_filter

    footprint = np.ones((2 * neigh_f + 1, 2 * neigh_t + 1))
    local_max = maximum_filter(S, footprint=footprint) == S
    floor = np.percentile(S, amp_pct)
    detected = local_max & (S > floor)
    f_idx, t_idx = np.where(detected)
    peaks = list(zip(t_idx.tolist(), f_idx.tolist()))
    peaks.sort()
    return peaks

def generate_hashes(peaks, fan=FAN_VALUE):
    hashes = []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        for j in range(1, fan + 1):
            if i + j >= n:
                break
            t2, f2 = peaks[i + j]
            dt = t2 - t1
            if MIN_DT <= dt <= MAX_DT:
                h = hashlib.sha1(f"{f1}|{f2}|{dt}".encode()).hexdigest()[:20]
                hashes.append((h, t1))
    return hashes

def fingerprint_samples(samples):
    S, _, _, _ = compute_spectrogram(samples)
    peaks = find_peaks(S)
    return generate_hashes(peaks), peaks, S

def fingerprint_file(path):
    return fingerprint_samples(load_audio(path))

class FingerprintDB:

    def __init__(self):
        self.index = defaultdict(list)
        self.songs = {}

    def add_song(self, label, hashes):
        for h, t in hashes:
            self.index[h].append((label, t))
        self.songs[label] = self.songs.get(label, 0) + len(hashes)

    def build_from_folder(self, folder, exts=(".mp3", ".wav", ".flac", ".ogg", ".m4a")):
        files = []
        for e in exts:
            files += glob.glob(os.path.join(folder, f"*{e}"))
        files = sorted(set(files))
        for f in files:
            label = os.path.splitext(os.path.basename(f))[0]
            hashes, _, _ = fingerprint_file(f)
            self.add_song(label, hashes)
        return len(files)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({"index": dict(self.index), "songs": self.songs}, fh)

    @classmethod
    def load(cls, path):
        db = cls()
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        db.index = defaultdict(list, data["index"])
        db.songs = data["songs"]
        return db

def match(query_hashes, db, return_details=False):
    offset_counts = defaultdict(lambda: defaultdict(int))
    for h, qt in query_hashes:
        for (label, dbt) in db.index.get(h, []):
            offset_counts[label][dbt - qt] += 1

    scores = {}
    best_offset_per_song = {}
    for label, offs in offset_counts.items():
        best_off, best_cnt = max(offs.items(), key=lambda kv: kv[1])
        scores[label] = best_cnt
        best_offset_per_song[label] = best_off

    if not scores:
        return (None, 0, {}, {}) if return_details else (None, 0, {})

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]

    if return_details:
        details = {
            "ranked": ranked,
            "best_offset": best_offset_per_song[best_label],
            "offset_hist": dict(offset_counts[best_label]),
        }
        return best_label, best_score, scores, details
    return best_label, best_score, scores

def match_single_peaks(query_peaks, db_single_index):
    scores = defaultdict(int)
    qfreqs = set(f for _, f in query_peaks)
    for label, freqset in db_single_index.items():
        scores[label] = len(qfreqs & freqset)
    if not scores:
        return None, 0, {}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0], ranked[0][1], dict(scores)

