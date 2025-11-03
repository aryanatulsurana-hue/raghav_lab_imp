import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time
from datetime import datetime

# ==== CONFIG ====
SAMPLE_RATE = 44100
CHUNK_DURATION = 0.01   

N_MELS = 64
FMIN, FMAX = 500, 8000
WINDOW_SIZE = 0.01
HOP_SIZE = 0.005
MODEL_PATH = "yellow184syllable_a_i_not_detector_new.h5"
LOG_FILE = "log_mel_spectrogram.txt"

CONF_THRESHOLD = 0.463
INTRO_CLASS = 0
A_CLASS = 1
INTRO_COOLDOWN = 0.060  # 60 ms
A_COOLDOWN = 0.150      # 150 ms
CONSEC_WINDOW = 0.060   # 35 ms

# ==== DERIVED PARAMETERS ====
chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)   # samples per callback

n_fft = int(WINDOW_SIZE * SAMPLE_RATE)
hop_length = int(HOP_SIZE * SAMPLE_RATE)

# Safety: ensure n_fft >= 1 etc.
n_fft = max(1, n_fft)
hop_length = max(1, hop_length)
chunk_samples = max(1, chunk_samples)

# ==== LOAD MODEL ====
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded:", model.input_shape, "→", model.output_shape)

# ==== DETECTION STATE ====
prev_confident_class = None
prev_confident_ts = 0.0
cooldown_end = 0.0

# ==== LOG FILE ====
f = open(LOG_FILE, "w")

# ==== FUNCTIONS ====
def mel_framewise_prob_avg(chunk):
    """
    Compute log-Mel spectrogram for chunk, convert to linear power,
    normalize each column (frame) so it sums to 1, then average columns
    to yield a single (N_MELS,) vector whose elements sum to 1.
    """
    # compute mel spectrogram (shape: n_mels x n_frames)
    S = librosa.feature.melspectrogram(
        y=chunk,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    # if S is empty (possible for extremely short chunk), create small epsilon matrix
    if S.size == 0:
        S = np.ones((N_MELS, 1)) * 1e-8

    # log scale then back to linear (matches training pipeline you used)
    logS = librosa.power_to_db(S, ref=np.max)          # shape (n_mels, n_frames)
    lin = librosa.db_to_power(logS, ref=1.0)          # linear power (n_mels, n_frames)

    # Ensure non-negative (lin should be >=0 but numerical safety)
    lin = np.maximum(lin, 0.0)

    # Frame-wise normalization: divide each column by its column-sum
    col_sums = np.sum(lin, axis=0, keepdims=True)    # shape (1, n_frames)
    # avoid divide by zero: if column sum is 0 -> replace with tiny epsilon
    col_sums = col_sums + 1e-12

    norm_frames = lin / col_sums                       # each column sums to 1

    # Debug check (can be commented out later)
    # print("DEBUG: column sums (should be 1):", np.sum(norm_frames, axis=0))

    # Average across time frames to make single 64-d vector (matching training)
    vec = np.mean(norm_frames, axis=1)                # shape (n_mels,)

    # final safety: ensure non-negative and sum to 1
    vec = np.maximum(vec, 0.0)
    s = vec.sum() + 1e-12
    vec = vec / s

    return vec, logS, norm_frames

def detect_syllable(probs, record_time):
    """Detect consecutive confident predictions and apply cooldowns."""
    global prev_confident_class, prev_confident_ts, cooldown_end

    if record_time < cooldown_end:
        return None

    confident_classes = np.where(probs > CONF_THRESHOLD)[0]
    if len(confident_classes) == 0:
        prev_confident_class = None
        return None

    cls = confident_classes[0]

    if prev_confident_class == cls and (record_time - prev_confident_ts) <= CONSEC_WINDOW:
        if cls == INTRO_CLASS:
            cooldown_end = record_time + INTRO_COOLDOWN
            event = "Intro note detected"
        elif cls == A_CLASS:
            cooldown_end = record_time + A_COOLDOWN
            event = "Syllable A detected"
        else:
            event = None

        prev_confident_class = None
        prev_confident_ts = 0.0
        return event

    prev_confident_class = cls
    prev_confident_ts = record_time
    return None

# ==== CALLBACK ====
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Input status:", status)

    record_time = time.time()
    chunk = indata[:, 0].copy()   # ensure copy, indata buffer may be reused

    audio_mean = np.mean(chunk)
    audio_std = np.std(chunk)

    vec, logS, norm_frames = mel_framewise_prob_avg(chunk)

    inp = vec.reshape(1, N_MELS, 1)
    probs = model.predict(inp, verbose=0)[0]

    event = detect_syllable(probs, record_time)

    write_time = time.time()
    latency_ms = (write_time - record_time) * 1000.0
    record_str = datetime.fromtimestamp(record_time).strftime("%H:%M:%S.%f")[:-3]
    write_str = datetime.fromtimestamp(write_time).strftime("%H:%M:%S.%f")[:-3]

    # === Only log when "a" or "i" probability exceeds 0.2 ===
    if probs[A_CLASS] > 0.2 or probs[INTRO_CLASS] > 0.2:
        log_line = (
            f"{record_str} {write_str} {latency_ms:.2f}ms "
            + " ".join(f"{p:.6f}" for p in probs)
        )
        if event:
            log_line += f" | {event}"

        f.write(log_line + "\n")
        f.flush()

        print(
            f"[Recorded {record_str} | Written {write_str} | Latency {latency_ms:.2f} ms] "
            + " ".join(f"{p:.3f}" for p in probs)
            + (f" | {event}" if event else "")
        )
"))


# ==== START STREAM ====
print("Recording from microphone... Press Ctrl+C to stop.")
with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=chunk_samples, channels=1, callback=audio_callback):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped recording.")
        f.close()
