import numpy as np
import librosa  # type: ignore
import scipy.io
import glob
import os
import matplotlib.pyplot as plt

# ================= PARAMETERS =================
N_MELS = 64
FMIN = 500             # Hz
FMAX = 8000            # Hz
BATCH_SIZE = 16

SAMPLE_RATE = 41000
WINDOW_SIZE = 0.002   # 2 ms
HOP_SIZE = 0.004      # 4 ms


def load_annotations(mat_file):
    mat = scipy.io.loadmat(mat_file)
    onsets = mat['onsets'].flatten()
    offsets = mat['offsets'].flatten()
    labels_raw = mat['labels']
    labels_str = labels_raw[0] if labels_raw.ndim > 0 else labels_raw
    labels = list(labels_str)
    return onsets, offsets, labels


def extract_logmel(wav_file):
    """
    Extract log-Mel spectrogram and normalize each frame to be probabilistic:
    values ∈ [0,1], sum = 1
    """
    y, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
    n_fft = int(WINDOW_SIZE * sr)
    hop_length = int(HOP_SIZE * sr)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=n_fft,
        hop_length=hop_length, fmin=FMIN, fmax=FMAX
    )
    logS = librosa.power_to_db(S, ref=np.max)

    # Probabilistic normalization per frame
    lin_mel = librosa.db_to_power(logS, ref=1.0)  # convert dB to linear power

    if lin_mel.shape[1] == 1:
        vec = lin_mel[:, 0]
        norm_vec = vec / np.sum(vec)
        logS_norm = norm_vec.reshape(-1, 1)
    else:
        logS_norm = lin_mel / np.sum(lin_mel, axis=0, keepdims=True)

    return logS_norm.T  # frames x n_mels




def create_label_vector_4_classes(onsets, offsets, labels, total_frames, hop_length):
    label_vec = 3 * np.ones(total_frames, dtype=int)   # 3 as unassigned/background
    regions = []

    for onset, offset, label in zip(onsets, offsets, labels):
        start_frame = int(np.floor(onset*32 / hop_length))
        end_frame = int(np.ceil(offset*32 / hop_length))
        regions.append((start_frame, end_frame, label))

        if label == 'a':
            label_vec[start_frame:end_frame+1] = 1
        elif label == 'i':
            label_vec[start_frame:end_frame+1] = 0

    # Fill noise (2) in gaps between syllables
    if regions:
        regions_sorted = sorted(regions, key=lambda x: x[0])
        for i in range(len(regions_sorted)-1):
            prev_end = regions_sorted[i][1]
            next_start = regions_sorted[i+1][0]
            if prev_end+1 < next_start:
                label_vec[prev_end+1:next_start] = 2
        first_start = regions_sorted[0][0]
        if first_start > 0:
            label_vec[:first_start] = 2
        last_end = regions_sorted[-1][1]
        if last_end < total_frames-1:
            label_vec[last_end+1:] = 2
    else:
        label_vec[:] = 2

    return label_vec




def find_onset_cluster(onset_indices, frame_duration_sec, cluster_size=2, window_ms=8):
    window_sec = window_ms / 1000.0
    n = len(onset_indices)
    for i in range(n - cluster_size + 1):
        start = onset_indices[i]
        end = onset_indices[i + cluster_size - 1]
        time_diff = (end - start) * frame_duration_sec
        if time_diff <= window_sec:
            # return the start frame of the cluster
            return start
    return None

from tensorflow.keras.callbacks import LearningRateScheduler #type: ignore

def step_decay(epoch):
    initial_lr = 1e-3
    drop = 0.5             # decay factor
    epochs_drop = 100       # decay every 30 epochs
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    print(f"Epoch {epoch+1}: Learning rate is {lr:.6f}")
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

def load_dataset_4_classes_global_balanced(data_dir):
    wav_files = sorted(glob.glob(os.path.join(data_dir, '*.wav')))
    mat_files = sorted(glob.glob(os.path.join(data_dir, '*.not.mat')))
    print(f"Found {len(wav_files)} wav files and {len(mat_files)} mat files.")

    all_logmel = []
    all_y = []
    all_indices = {0: [], 1: [], 2: [], 3: []}

    for wav_file, mat_file in zip(wav_files, mat_files):
        logmel = extract_logmel(wav_file)
        hop_length = 128
        onsets, offsets, labels = load_annotations(mat_file)
        label_vec = create_label_vector_4_classes(onsets, offsets, labels, logmel.shape[0], hop_length)

        all_logmel.append(logmel)
        all_y.append(label_vec)

    if len(all_logmel) > 0:
        all_logmel_concat = np.concatenate(all_logmel, axis=0)
    else:
        all_logmel_concat = np.array([])

    if len(all_y) > 0:
        all_y_concat = np.concatenate(all_y, axis=0)
        print(f"Concatenated label vector shape: {all_y_concat.shape}")
    else:
        all_y_concat = np.array([])

    for cls in [0, 1, 2, 3]:
        cls_indices = np.where(all_y_concat == cls)[0]
        all_indices[cls] = cls_indices
        print(f"Class {cls} has {len(cls_indices)} samples")

    # Balance dataset
    class_counts = [len(all_indices[cls]) for cls in [0, 1, 2, 3]]
    sorted_counts = sorted(class_counts, reverse=True)
    target_size = sorted_counts[1]  # second highest
    np.random.seed(42)
    balanced_indices = []

    for cls in [0, 1, 2, 3]:
        idxs = all_indices[cls]
        if len(idxs) >= target_size:
            selected = np.random.choice(idxs, target_size, replace=False)
        else:
            selected = np.random.choice(idxs, target_size, replace=True)
        balanced_indices.extend(selected)
    balanced_indices.sort()

    X_balanced = all_logmel_concat[balanced_indices]
    y_balanced = all_y_concat[balanced_indices]
    y_balanced = y_balanced.reshape(-1, 1)

    return X_balanced, y_balanced







from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def build_model_4_classes(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),

        Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),

        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(4, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(0.001),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


from tensorflow.keras.utils import plot_model

model = build_model_4_classes((64, 1))
plot_model(model, show_shapes=True, show_layer_names=True, to_file="model_architecture.png")


import numpy as np

def get_predicted_frames_for_classes(pred_probs, threshold=0.65):
    pred_classes = []
    for probs in pred_probs:
        max_prob = np.max(probs)
        if max_prob > threshold:
            pred_class = np.argmax(probs)
        else:
            pred_class = -1  # no confident prediction
        pred_classes.append(pred_class)
    pred_classes = np.array(pred_classes)

    frames_of_a = np.where(pred_classes == 1)[0].tolist()  # all frames predicted as class 1 ('a')
    frames_of_i = np.where(pred_classes == 0)[0].tolist()  # all frames predicted as class 0 ('i')

    return frames_of_a, frames_of_i

     
    








def get_true_intro_segments(labels, onsets, offsets):
    true_segs = [(int(onsets[i]), int(offsets[i])) for i, l in enumerate(labels) if l == 'i']
    return true_segs


def true_intro_notes_before_a(labels, onsets, offsets):
    a_indices = [i for i, l in enumerate(labels) if l == 'a']
    if not a_indices:
        return 0
    first_a = a_indices[0]
    count_intro = sum(1 for i, l in enumerate(labels[:first_a]) if l == 'i')
    return count_intro


def filter_pred_intro_segs_before_a(intro_segs_pred, a_onset_idx, hop_length_samples):
    a_frame_idx = int(a_onset_idx / hop_length_samples)
    filtered_segs = [seg for seg in intro_segs_pred if seg[1] < a_frame_idx]
    return filtered_segs


def plot_onset_clusters(logmel_data, pred_a_onsets, pred_i_onsets, hop_length_samples, window_ms=9):
    fs = SAMPLE_RATE
    hop_sec = hop_length_samples / fs
    loudness = logmel_data.max(axis=1)
    times = np.arange(len(loudness)) * hop_sec

    plt.figure(figsize=(14, 5))
    plt.plot(times, loudness, color='gray', label='Loudness (max Mel bin)')

    # Plot a onsets as green vertical lines
    for onset in pred_a_onsets:
        plt.axvline(onset * hop_sec, color='green', linestyle='-', linewidth=1, alpha=0.7, label='Predicted a onset' if onset == pred_a_onsets[0] else "")

    # Plot i onsets as blue vertical lines
    for onset in pred_i_onsets:
        plt.axvline(onset * hop_sec, color='blue', linestyle='-', linewidth=1, alpha=0.7, label='Predicted i onset' if onset == pred_i_onsets[0] else "")

    plt.xlabel('Time (seconds)')
    plt.ylabel('Loudness (max Mel bin)')
    plt.title(f'Predicted "a" and "i" onsets clusters (window={window_ms} ms)')
    plt.legend()
    plt.grid(True)
    plt.show()

    import numpy as np

import os
import glob
import numpy as np

def compute_intro_note_stats_from_folder(folder_path, intro_label):
    annotation_files = sorted(glob.glob(os.path.join(folder_path, '*.not.mat')))
    all_durations = []

    for mat_file in annotation_files:
        onsets, offsets, labels = load_annotations(mat_file)

        intro_durations = [
            offset - onset
            for onset, offset, label in zip(onsets, offsets, labels)
            if label == intro_label
        ]

        all_durations.extend(intro_durations)

    all_durations = np.array(all_durations)
    avg_length = np.mean(all_durations) if all_durations.size > 0 else 0
    variance_length = np.var(all_durations) if all_durations.size > 0 else 0

    return avg_length, variance_length





import matplotlib.pyplot as plt
import numpy as np
def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    from tensorflow.keras.utils import to_categorical

    data_dir = 'training-data/yellow184pink176/WavFiles/Wav-files-note-files-paired'
    X, y = load_dataset_4_classes_global_balanced(data_dir)

    print(f'Training data shape: {X.shape}, labels shape: {y.shape}')
    X = X.astype(np.float32)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    y_cat = to_categorical(y, num_classes=4)
    print("One-hot encoded labels shape:", y_cat.shape)
    print("Sample one-hot labels (first 5):\n", y_cat[:5])

    # Split for validation
    X_train, X_val, y_train_cat, y_val_cat = train_test_split(X, y_cat, test_size=0.1, random_state=42)

    # Optional: StandardScaler for X
    X_train_reshaped = X_train.reshape(-1, 64)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(-1, 64, 1)

    model = build_model_4_classes((X.shape[1], 1))

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1,2,3]), y=y.flatten())
    class_weights_dict = {i: class_weights[i] for i in range(4)}
    print("Class weights:", class_weights_dict)

    # Train model
    history = model.fit(
        X, y_cat, batch_size=BATCH_SIZE, epochs=1250,
        shuffle=True, class_weight=class_weights_dict,
        validation_split=0.1, callbacks=[lr_scheduler]
    )

    # Evaluate
    y_val_pred_prob = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    y_true = np.argmax(y_val_cat, axis=1)

    cm = confusion_matrix(y_true, y_val_pred)
    print("Confusion matrix:\n", cm)
    acc = accuracy_score(y_true, y_val_pred)
    print(f'Validation accuracy: {acc:.4f}')
    print(classification_report(y_true, y_val_pred))

    model.save('yellow184syllable_a_i_not_detector.h5')
    print('Model saved as yellow184syllable_a_i_not_detector.h5')

    # ================= PLOTS =================
    plt.figure(figsize=(14, 7))

    # Loss
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss', color='cyan')

    # Precision per class
    for cls in range(4):
        key = f'precision_class_{cls}' if f'precision_class_{cls}' in history.history else f'precision_{cls}'
        val_key = f'val_precision_class_{cls}' if f'val_precision_class_{cls}' in history.history else None
        plt.plot(history.history[key], linestyle='--', label=f'Precision class {cls}')
        if val_key and val_key in history.history:
            plt.plot(history.history[val_key], linestyle=':', label=f'Val Precision class {cls}')

    # Recall per class
    for cls in range(4):
        key = f'recall_class_{cls}' if f'recall_class_{cls}' in history.history else f'recall_{cls}'
        val_key = f'val_recall_class_{cls}' if f'val_recall_class_{cls}' in history.history else None
        plt.plot(history.history[key], linestyle=':', label=f'Recall class {cls}')
        if val_key and val_key in history.history:
            plt.plot(history.history[val_key], linestyle='-.', label=f'Val Recall class {cls}')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training & Validation Loss, Precision, Recall per Class')
    plt.legend()
    plt.grid(True)
    plt.show()
    
'''

    def detect_segment_switch(pred_a_onsets, pred_i_onsets):
        pred_a_times = [frame * 0.004 for frame in pred_a_onsets]
        pred_i_times = [frame * 0.004 for frame in pred_i_onsets]
        print('Predicted onsets of a (seconds):', pred_a_times)
        print('Predicted onsets of i (seconds):', pred_i_times)

        i_starts = []
        a_starts = []
        last_i_time = -float('inf')
        last_a_time = -float('inf')
        idx_i = 0
        idx_a = 0
        N = len(pred_i_times)
        M = len(pred_a_times)

        while idx_i < N or idx_a < M:
            # Check i start condition if frames remain
            if idx_i + 1 < N:
                if pred_i_times[idx_i + 1] - pred_i_times[idx_i] <= 0.009 and \
                pred_i_times[idx_i] > last_i_time + 0.075:
                    i_starts.append(pred_i_times[idx_i])
                    last_i_time = pred_i_times[idx_i]
                    idx_i += 1
                else:
                    idx_i += 1
            else:
                idx_i += 1  # Move index to break the loop eventually

            # Check a cluster condition if frames remain
            if idx_a + 2 < M:
                if pred_a_times[idx_a + 2] - pred_a_times[idx_a] <= 0.017 and \
                pred_a_times[idx_a] > last_a_time + 0.150:
                    a_starts.append(pred_a_times[idx_a])
                    last_a_time = pred_a_times[idx_a]
                    idx_a += 3  # Skip cluster frames already counted
                else:
                    idx_a += 1
            else:
                idx_a += 1  # Move index if no cluster detected or at frame end

        return i_starts, a_starts




    def print_visual_output(i_starts, a_detected, a_start):
        print("Detected 'i' segment starts (s):", i_starts)
        if a_detected:
            print("Detected 'a' segment switch at (s):", a_start)
        else:
            print("'a' segment switch not detected.")

    # ... continue with your testing as before ...


    # === TESTING ===
    test_wav = 'training-data/yellow184pink176/WavFiles/Wav-files-note-files-paired/yellow184pink176_yellow158red176_030624073914.wav'
    test_mat = 'training-data/yellow184pink176/WavFiles/Wav-files-note-files-paired/yellow184pink176_yellow158red176_030624073914.wav.not.mat'
    logmel_test = extract_logmel(test_wav)
    X_test = np.expand_dims(logmel_test.astype(np.float32), axis=2)
    import tensorflow as tf #type:ignore
    loaded_model = tf.keras.models.load_model('yellow184syllable_a_i_not_detector.h5')
    pred_probs = loaded_model.predict(X_test)
    
    

# Optionally, set frames to -1 if not confident (threshold)
    threshold = 0.463
    confident_labels = np.where(pred_probs.max(axis=1) > threshold,
                            np.argmax(pred_probs, axis=1),
                            -1)

    onsets, offsets, labels = load_annotations(test_mat)
    #print(f"Lengths - onsets: {len(onsets)}, offsets: {len(offsets)}, labels: {len(labels)}")
    logmel = extract_logmel(test_wav)
    hop_length = 128
    
    actual_labels = create_label_vector_4_classes(onsets, offsets, labels, logmel.shape[0], hop_length)
    #predicted_labels = np.argmax(pred_probs, axis=1)  # shape: (frames,)
    unique_vals, counts = np.unique(actual_labels, return_counts=True)
    for val, count in zip(unique_vals, counts):
        print(f"Class {val}: {count} frames")
    print(actual_labels)
    a_predicted_stamps =[]
    i_predicted_stamps =[]

    for idx, item in enumerate(actual_labels):
        if item == 1:
            a_predicted_stamps.append(idx)

        if item == 0:
            i_predicted_stamps.append(idx)


    frame_duration = 0.004  # 4ms per frame
    N_frames = len(confident_labels)
    import numpy as np
    import matplotlib.pyplot as plt

    class_labels = [0, 1, 2, 3]
    thresholds = np.linspace(0.25, 0.8, 50)
    scores = []

    for thresh in thresholds:
        confident_labels = np.where(pred_probs.max(axis=1) > thresh,
                               np.argmax(pred_probs, axis=1),
                               -1)
    # Only keep valid class predictions
        pred_counts = np.array([(confident_labels == c).sum() for c in class_labels])
        actual_counts = np.array([(actual_labels == c).sum() for c in class_labels])
    # To prevent division by zero, replace zero predictions with 1
        safe_pred_counts = np.where(pred_counts != 0, pred_counts, 1)
        error_ratio = np.abs(actual_counts - pred_counts) / safe_pred_counts
        score = np.sum(error_ratio)
        scores.append(score)

    scores = np.array(scores)
    best_idx = np.argmin(scores)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold (lowest score): {best_threshold:.3f} with score: {scores[best_idx]:.3f}")

    plt.figure(figsize=(10,5))
    plt.plot(thresholds, scores, marker='o')
    plt.scatter([best_threshold], [scores[best_idx]], color='red', label='Best threshold')
    plt.title("Summation of per-class absolute error ratio vs Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel(f'$\\sum_{'c'} \left| (A_c - P_c)/P_c \\right|$')
    plt.legend()
    plt.grid(True)
    plt.show()

    time = np.arange(N_frames) * frame_duration

    class_names = {0: 'i', 1: 'a', 2: 'noise', 3: 'othersyllables'}
    colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red'}

# Plot: Time vs Predictions and Time vs Actual
    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'hspace': 0.2})
    for cls in range(4):
        axs[0].scatter(time[confident_labels == cls], [cls] * np.sum(confident_labels == cls),
                  color=colors[cls], label=f"Pred {class_names[cls]}", s=12, alpha=0.8, marker='o')
        axs[1].scatter(time[actual_labels == cls], [cls] * np.sum(actual_labels == cls),
                  color=colors[cls], label=f"True {class_names[cls]}", s=12, alpha=0.8, marker='x')

    axs[0].set_title("Time vs Predicted Class")
    axs[1].set_title("Time vs Actual (Ground Truth) Class")
    for ax in axs:
        ax.set_yticks(range(4))
        ax.set_yticklabels([class_names[c] for c in range(4)])
        ax.set_ylabel("Class")
        ax.grid(True)
    axs[1].set_xlabel("Time (s)")
    axs[0].legend(loc='upper right', ncol=2)
    axs[1].legend(loc='upper right', ncol=2)
    plt.tight_layout()
    plt.show()

# Print frame counts for each class
    for cls in range(4):
        pred_count = np.sum(confident_labels == cls)
        true_count = np.sum(actual_labels == cls)
        print(f"Class {class_names[cls]}: Predicted {pred_count} frames, Actual {true_count} frames")
    
   

    
    i_starts, a_starts = detect_segment_switch(a_predicted_stamps, i_predicted_stamps)
    
    import matplotlib.pyplot as plt

    def print_visual_output(i_starts, a_starts, actual_onsets, actual_labels, frame_duration=0.004):
        plt.figure(figsize=(12, 3))

        # Plot predicted i starts
        plt.scatter(i_starts, [0]*len(i_starts), color='blue', label='Predicted i starts', marker='o', s=70)

        # Plot detected a start at y=1 if present
        if a_starts is not None:
            plt.scatter([a_starts], [0]*len(a_starts), color='orange', label='Detected a start', marker='x', s=120)

        # Convert actual onsets from frames to seconds
        actual_times = actual_onsets/1000 

        # Separate actual onsets into 'a' and 'i'
        actual_a_times = [t for t, lab in zip(actual_times, actual_labels) if lab == 'a']
        actual_i_times = [t for t, lab in zip(actual_times, actual_labels) if lab == 'i']

        # Plot actual onsets
        plt.scatter(actual_i_times, [-0.3]*len(actual_i_times), color='cyan', label='Actual i onsets', marker='^', s=60)
        plt.scatter(actual_a_times, [1.3]*len(actual_a_times), color='magenta', label='Actual a onsets', marker='^', s=60)

        # Setup y-axis ticks and limits
        plt.yticks([0, 1], ['Predicted i starts', 'Detected a start'])
        plt.ylim(-1, 2)
        plt.xlabel('Time (seconds)')
        plt.title('Predicted and Actual Onsets of i and a syllables')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
    print_visual_output(i_starts, a_starts, onsets, labels)'''




    

if __name__ == "__main__":
    main()



