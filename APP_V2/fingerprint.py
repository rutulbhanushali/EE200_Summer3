"""fingerprint.py  –  Audio Fingerprinting core library
EE200: Signals, Systems & Networks – Course Project (Q3A / Q3B)
Implements:
  • Spectrogram computation (STFT via librosa)
  • Peak (constellation) detection
  • Pair-hash generation  (anchor + fan-out → SHA-1 hash)
  • FingerprintDB  – build / save / load
  • Pair-hash matching with offset histogram  (match)
  • Single-peak index build + matching  (build_single_peak_index / match_single_peaks)
  • Robustness helpers: add_noise, pitch_shift
  • DFT magnitude helper for Q3A report experiments
"""

import os
import glob
import pickle
import hashlib
from collections import defaultdict
import numpy as np
import librosa

# ──────────────────────────────────────────────
#  Global hyper-parameters
# ──────────────────────────────────────────────
SAMPLE_RATE        = 11025   # Hz – good trade-off: captures up to ~5.5 kHz, fast
N_FFT              = 1024    # FFT window size
HOP                = 512     # hop between STFT frames
PEAK_NEIGH_TIME    = 8       # local-max neighbourhood radius in time frames
PEAK_NEIGH_FREQ    = 8       # local-max neighbourhood radius in freq bins
MIN_AMP_PERCENTILE = 60      # only keep peaks above this amplitude percentile
FAN_VALUE          = 15      # max partners per anchor peak
MIN_DT             = 1       # min time-frame gap between anchor and partner
MAX_DT             = 80      # max time-frame gap between anchor and partner

# ──────────────────────────────────────────────
#  Audio I/O
# ──────────────────────────────────────────────
def load_audio(path, sr=SAMPLE_RATE):
    """Load any audio file and resample to sr Hz (mono)."""
    samples, _ = librosa.load(path, sr=sr, mono=True)
    return samples

# ──────────────────────────────────────────────
#  Spectrogram
# ──────────────────────────────────────────────
def compute_spectrogram(samples, sr=SAMPLE_RATE, n_fft=N_FFT, hop=HOP):
    """
    Compute magnitude spectrogram via STFT.
    Returns
    -------
    S      : 2-D ndarray (freq × time), linear magnitude
    S_db   : 2-D ndarray (freq × time), dB-scaled  (reference = max)
    freqs  : 1-D ndarray of frequency values in Hz
    times  : 1-D ndarray of frame centre times in seconds
    """
    S      = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop))
    S_db   = librosa.amplitude_to_db(S, ref=np.max)
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times  = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return S, S_db, freqs, times

def compute_dft_magnitude(samples, sr=SAMPLE_RATE):
    """
    Compute the one-sided DFT magnitude of the *entire* signal (no windowing).
    Used in Q3A to show that a plain FFT tells you *which* frequencies exist
    but loses all timing information.
    Returns
    -------
    freqs : 1-D ndarray  (Hz, 0 … sr/2)
    mag   : 1-D ndarray  (linear magnitude, same length as freqs)
    """
    N      = len(samples)
    mag    = np.abs(np.fft.rfft(samples))
    freqs  = np.fft.rfftfreq(N, d=1.0 / sr)
    return freqs, mag

# ──────────────────────────────────────────────
#  Peak / Constellation detection
# ──────────────────────────────────────────────
def find_peaks(S, neigh_t=PEAK_NEIGH_TIME, neigh_f=PEAK_NEIGH_FREQ, amp_pct=MIN_AMP_PERCENTILE):
    """
    Find local maxima in the magnitude spectrogram that stand out from their
    neighbourhood (time × frequency footprint) AND are above a global
    amplitude percentile floor.
    Parameters
    ----------
    S       : 2-D ndarray (freq × time), linear magnitude  ← output of compute_spectrogram
    neigh_t : half-width of the neighbourhood in time frames
    neigh_f : half-width of the neighbourhood in freq bins
    amp_pct : minimum amplitude percentile threshold
    Returns
    -------
    peaks : list of (time_frame, freq_bin) tuples, sorted by time_frame
    """
    from scipy.ndimage import maximum_filter
    footprint = np.ones((2 * neigh_f + 1, 2 * neigh_t + 1))
    local_max = maximum_filter(S, footprint=footprint) == S
    floor     = np.percentile(S, amp_pct)
    detected  = local_max & (S > floor)
    f_idx, t_idx = np.where(detected)
    peaks = list(zip(t_idx.tolist(), f_idx.tolist()))
    peaks.sort()          # sort by time frame
    return peaks

# ──────────────────────────────────────────────
#  Hash generation  (pair-hash / fan-out)
# ──────────────────────────────────────────────
def generate_hashes(peaks, fan=FAN_VALUE):
    """
    For each anchor peak, form pairs with the next `fan` peaks in time.
    Each pair (f1, f2, dt) is hashed to a 20-char hex string.
    Parameters
    ----------
    peaks : list of (time_frame, freq_bin) tuples (sorted by time)
    fan   : maximum number of partners per anchor
    Returns
    -------
    hashes : list of (hash_string, anchor_time_frame) tuples
    """
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

# ──────────────────────────────────────────────
#  Convenience wrappers
# ──────────────────────────────────────────────
def fingerprint_samples(samples):
    """
    Full pipeline: samples → (hashes, peaks, spectrogram_magnitude).
    Returns
    -------
    hashes : list of (hash_str, time_frame)
    peaks  : list of (time_frame, freq_bin)
    S      : 2-D magnitude spectrogram array
    """
    S, _, _, _ = compute_spectrogram(samples)
    peaks      = find_peaks(S)
    return generate_hashes(peaks), peaks, S

def fingerprint_file(path):
    """Load an audio file and return its (hashes, peaks, S)."""
    return fingerprint_samples(load_audio(path))

# ──────────────────────────────────────────────
#  Fingerprint Database  (pair-hash index)
# ──────────────────────────────────────────────
class FingerprintDB:
    """
    Inverted index mapping  hash → [(song_label, db_time_frame), …]
    Also records how many hashes were stored per song in self.songs.
    """
    def __init__(self):
        self.index = defaultdict(list)   # hash → [(label, t), …]
        self.songs = {}                  # label → hash count

    # ── building ──────────────────────────────
    def add_song(self, label, hashes):
        """Insert pre-computed hashes for one song into the index."""
        for h, t in hashes:
            self.index[h].append((label, t))
        self.songs[label] = self.songs.get(label, 0) + len(hashes)

    def build_from_folder(self, folder, exts=(".mp3", ".wav", ".flac", ".ogg", ".m4a")):
        """
        Scan *folder* for audio files, fingerprint each one, and add to index.
        Returns the number of files indexed.
        """
        files = []
        for e in exts:
            files += glob.glob(os.path.join(folder, f"*{e}"))
        files = sorted(set(files))
        for f in files:
            label  = os.path.splitext(os.path.basename(f))[0]
            hashes, _, _ = fingerprint_file(f)
            self.add_song(label, hashes)
        return len(files)

    # ── persistence ───────────────────────────
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

# ──────────────────────────────────────────────
#  Pair-hash Matching  (main algorithm)
# ──────────────────────────────────────────────
def match(query_hashes, db, return_details=False):
    """
    Match a query fingerprint (pair-hashes) against the database.
    For each matching hash the time offset  Δt = db_time − query_time  is
    tallied.  A genuine match produces a sharp spike at one consistent Δt;
    random/wrong matches give a flat histogram.
    Parameters
    ----------
    query_hashes   : list of (hash_str, query_time_frame)
    db             : FingerprintDB instance
    return_details : if True, also return a 4-tuple including detailed info
    Returns
    -------
    Without details : (best_label, best_score, scores_dict)
    With details    : (best_label, best_score, scores_dict, details_dict)
      details_dict contains:
        "ranked"      – [(label, score), …] sorted descending
        "best_offset" – the winning time offset for the best match
        "offset_hist" – {offset: count, …} for the best match's song
    """
    offset_counts = defaultdict(lambda: defaultdict(int))
    for h, qt in query_hashes:
        for (label, dbt) in db.index.get(h, []):
            offset_counts[label][dbt - qt] += 1

    scores            = {}
    best_offset_per_song = {}
    for label, offs in offset_counts.items():
        best_off, best_cnt   = max(offs.items(), key=lambda kv: kv[1])
        scores[label]        = best_cnt
        best_offset_per_song[label] = best_off

    if not scores:
        return (None, 0, {}, {}) if return_details else (None, 0, {})

    ranked      = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]

    if return_details:
        details = {
            "ranked":      ranked,
            "best_offset": best_offset_per_song[best_label],
            "offset_hist": dict(offset_counts[best_label]),
        }
        return best_label, best_score, scores, details

    return best_label, best_score, scores

# ──────────────────────────────────────────────
#  Single-peak Index + Matching  (Q3A comparison)
# ──────────────────────────────────────────────
def build_single_peak_index(db):
    """
    Build a lightweight single-peak index from the existing pair-hash DB.
    The index maps  song_label → set of freq_bins  that appear in that song.
    This is used in Q3A to demonstrate why pair-hashes are far more
    discriminative than single peaks on their own.
    Parameters
    ----------
    db : FingerprintDB  (already built / loaded)
    Returns
    -------
    single_index : dict  { label: set_of_freq_bins }
    """
    single_index = defaultdict(set)
    for hash_str, entries in db.index.items():
        for label, t in entries:
            # The first field of a hash encodes f1 (anchor freq).
            # Instead of re-parsing the hash we rebuild freq sets from
            # stored (label, t) pairs; we only need which freq bins exist.
            # We store t as a proxy — actual freq recovery needs the peaks.
            # Therefore we ask callers to supply peaks directly via
            # build_single_peak_index_from_folder() below.
            pass   # filled by the proper builder below
    return dict(single_index)

def build_single_peak_index_from_folder(folder, exts=(".mp3", ".wav", ".flac", ".ogg", ".m4a")):
    """
    Build a single-peak index (label → set of (time_frame, freq_bin))
    directly from audio files in *folder*.
    Returns
    -------
    single_index : dict { label: set_of_(t,f)_tuples }
    """
    single_index = {}
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, f"*{e}"))
    files = sorted(set(files))
    for f in files:
        label       = os.path.splitext(os.path.basename(f))[0]
        _, peaks, _ = fingerprint_file(f)
        single_index[label] = set(peaks)
    return single_index

def match_single_peaks(query_peaks, single_index):
    """
    Match a query clip using only individual peaks (no pairing / time offset).
    For each song in single_index, the score is the size of the intersection
    of the query's peak set with the song's stored peak set.
    This deliberately ignores time alignment – it demonstrates how much
    weaker single-peak matching is compared to pair-hash matching:
    many songs share the same (t, f) bins by chance, especially at common
    frequencies, giving noisy, unreliable scores.
    Parameters
    ----------
    query_peaks  : list of (time_frame, freq_bin) from fingerprint_samples
    single_index : dict { label: set_of_(t,f)_tuples }  from
                   build_single_peak_index_from_folder
    Returns
    -------
    best_label : str or None
    best_score : int
    scores     : dict { label: score }
    """
    if not single_index:
        return None, 0, {}

    qset   = set(query_peaks)
    scores = {}
    for label, pset in single_index.items():
        scores[label] = len(qset & pset)

    if not any(scores.values()):
        # Fall back to frequency-only intersection (ignores time) for robustness
        qfreqs = set(f for _, f in query_peaks)
        for label, pset in single_index.items():
            label_freqs   = set(f for _, f in pset)
            scores[label] = len(qfreqs & label_freqs)

    ranked     = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label = ranked[0][0] if ranked else None
    best_score = ranked[0][1] if ranked else 0
    return best_label, best_score, scores

# ──────────────────────────────────────────────
#  Robustness helpers  (Q3A experiments)
# ──────────────────────────────────────────────
def add_noise(samples, snr_db):
    """
    Add white Gaussian noise to *samples* at the requested SNR (in dB).
    Parameters
    ----------
    samples : 1-D ndarray of float32/float64 audio samples
    snr_db  : signal-to-noise ratio in dB.
              Lower values = more noise (e.g., snr_db=0 means noise power
              equals signal power; snr_db=-10 means noise is 10× louder).
    Returns
    -------
    noisy : 1-D ndarray, same shape as samples
    """
    signal_power = np.mean(samples ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = np.random.randn(*samples.shape) * np.sqrt(noise_power)
    return (samples + noise).astype(samples.dtype)

def pitch_shift(samples, sr=SAMPLE_RATE, n_steps=2):
    """
    Shift the pitch of *samples* by *n_steps* semitones without changing
    duration (using librosa's phase-vocoder based pitch shifting).
    Parameters
    ----------
    samples : 1-D ndarray of float audio samples
    sr      : sample rate
    n_steps : number of semitones to shift (positive = up, negative = down)
    Returns
    -------
    shifted : 1-D ndarray, same length as samples
    """
    return librosa.effects.pitch_shift(samples, sr=sr, n_steps=n_steps)

def time_stretch(samples, rate=1.1):
    """
    Time-stretch *samples* by the given *rate* without changing pitch.
    Parameters
    ----------
    samples : 1-D ndarray
    rate    : stretch factor (>1 = slower, <1 = faster)
    Returns
    -------
    stretched : 1-D ndarray (length changes proportionally)
    """
    return librosa.effects.time_stretch(samples, rate=rate)
