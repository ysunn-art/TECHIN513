"""
PsychoMouse Analysis Pipeline
===============================
Complete pipeline: Label → Signal Processing → Feature Extraction → ML Training → Evaluation

Usage:
    Step 1: Label all sessions
        python analyze.py label-all

    Step 2: Run full pipeline (processing + training + evaluation)
        python analyze.py run

    Step 3: (Optional) Visualize a single session
        python analyze.py visualize data/eason02/20260221_...

Requirements:
    pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib
"""

import os
import sys
import json
import csv
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
MODEL_DIR = Path("models")

# Signal Processing Parameters
BUTTERWORTH_ORDER = 4
BUTTERWORTH_CUTOFF_HZ = 20        # Low-pass filter cutoff
MIN_SAMPLING_RATE_HZ = 100        # Minimum acceptable rate

# FFT Parameters
FFT_WINDOW_SEC = 2.0              # FFT window size in seconds
TREMOR_BAND_LOW = 8               # Tremor band lower bound (Hz)
TREMOR_BAND_HIGH = 12             # Tremor band upper bound (Hz)

# Feature Extraction Parameters
WINDOW_SIZE_SEC = 30              # Sliding window for feature extraction
WINDOW_STEP_SEC = 15              # Step size (50% overlap)
IDLE_VELOCITY_THRESHOLD = 50      # px/s - below this is "idle/micro-motion"
PAUSE_THRESHOLD_MS = 500          # Pause = no movement for >500ms

# Labeling
# Sessions are ~60-75 min each
# Alert = first 10 min (freshest), Fatigued = after 40 min (fatigue onset)
ALERT_MINUTES = 10
FATIGUED_MINUTES = 40

# ============================================================
# Step 1: Data Loading & Labeling
# ============================================================

def find_all_sessions(data_dir=DATA_DIR):
    """Find all session directories that contain mouse_events.csv."""
    sessions = []
    for participant_dir in sorted(data_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        for session_dir in sorted(participant_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            csv_path = session_dir / "mouse_events.csv"
            if csv_path.exists():
                sessions.append({
                    "participant": participant_dir.name,
                    "session_id": session_dir.name,
                    "session_dir": session_dir,
                    "csv_path": csv_path,
                })
    return sessions


def load_session(csv_path, max_rows=None):
    """Load a session CSV into a DataFrame."""
    dtype = {
        "timestamp_ms": np.int64,
        "x": np.float64,
        "y": np.float64,
        "event_type": str,
    }
    df = pd.read_csv(csv_path, dtype=dtype, nrows=max_rows)
    return df


def assign_time_labels(df, alert_min=ALERT_MINUTES, fatigued_min=FATIGUED_MINUTES):
    """Assign Alert/Transition/Fatigued labels based on elapsed time."""
    t0 = df["timestamp_ms"].iloc[0]
    elapsed_ms = df["timestamp_ms"] - t0
    alert_ms = alert_min * 60 * 1000
    fatigued_ms = fatigued_min * 60 * 1000

    conditions = [
        elapsed_ms < alert_ms,
        (elapsed_ms >= alert_ms) & (elapsed_ms < fatigued_ms),
        elapsed_ms >= fatigued_ms,
    ]
    labels = ["Alert", "Transition", "Fatigued"]
    df["label"] = np.select(conditions, labels, default="Transition")
    return df


# ============================================================
# Step 2: Signal Processing
# ============================================================

def estimate_sampling_rate(timestamps_ms):
    """Estimate the effective sampling rate from timestamps."""
    dt = np.diff(timestamps_ms)
    dt = dt[dt > 0]  # remove zero intervals
    if len(dt) == 0:
        return 0
    median_dt_ms = np.median(dt)
    return 1000.0 / median_dt_ms


def resample_to_uniform(df_move, target_rate_hz=200):
    """
    Resample irregular mouse data to uniform time intervals.
    This is necessary for proper filtering and FFT.
    """
    t = df_move["timestamp_ms"].values
    x = df_move["x"].values
    y = df_move["y"].values

    if len(t) < 10:
        return None

    t0 = t[0]
    t_end = t[-1]
    duration_ms = t_end - t0
    if duration_ms <= 0:
        return None

    # Create uniform time grid
    dt_ms = 1000.0 / target_rate_hz
    t_uniform = np.arange(t0, t_end, dt_ms)

    if len(t_uniform) < 10:
        return None

    # Interpolate x and y
    x_uniform = np.interp(t_uniform, t, x)
    y_uniform = np.interp(t_uniform, t, y)

    return pd.DataFrame({
        "timestamp_ms": t_uniform,
        "x": x_uniform,
        "y": y_uniform,
    })


def apply_butterworth_filter(data, cutoff_hz, fs, order=BUTTERWORTH_ORDER):
    """Apply zero-phase Butterworth low-pass filter."""
    nyquist = fs / 2
    if cutoff_hz >= nyquist:
        cutoff_hz = nyquist * 0.9
    b, a = sp_signal.butter(order, cutoff_hz / nyquist, btype="low")
    return sp_signal.filtfilt(b, a, data)


def compute_derivatives(x, y, fs):
    """Compute velocity, acceleration, and jerk from filtered positions."""
    dt = 1.0 / fs

    # Velocity (first derivative)
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed = np.sqrt(vx**2 + vy**2)

    # Acceleration (second derivative)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    accel = np.sqrt(ax**2 + ay**2)

    # Jerk (third derivative)
    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    jerk = np.sqrt(jx**2 + jy**2)

    return speed, accel, jerk, vx, vy


def compute_fft_features(signal_segment, fs):
    """
    Compute FFT and extract tremor band features.
    Returns: tremor_energy_ratio, spectral_entropy, psd, freqs
    """
    n = len(signal_segment)
    if n < 16:
        return 0.0, 0.0, None, None

    # Apply Hanning window
    windowed = signal_segment * np.hanning(n)

    # FFT
    yf = np.abs(fft(windowed))[:n // 2]
    xf = fftfreq(n, 1.0 / fs)[:n // 2]

    # Power spectral density
    psd = yf ** 2
    total_power = np.sum(psd)

    if total_power == 0:
        return 0.0, 0.0, psd, xf

    # Tremor band energy ratio (8-12 Hz)
    tremor_mask = (xf >= TREMOR_BAND_LOW) & (xf <= TREMOR_BAND_HIGH)
    tremor_power = np.sum(psd[tremor_mask])
    tremor_ratio = tremor_power / total_power

    # Spectral entropy
    psd_norm = psd / total_power
    psd_norm = psd_norm[psd_norm > 0]
    spec_entropy = entropy(psd_norm)

    return tremor_ratio, spec_entropy, psd, xf


# ============================================================
# Step 3: Feature Extraction
# ============================================================

def compute_path_efficiency(x, y):
    """Ratio of straight-line distance to actual path distance."""
    if len(x) < 2:
        return 1.0
    straight = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    actual = np.sum(diffs)
    if actual == 0:
        return 1.0
    return min(straight / actual, 1.0)


def count_pauses(speed, fs, threshold_ms=PAUSE_THRESHOLD_MS):
    """Count number of pauses (speed near zero for > threshold)."""
    min_samples = int(threshold_ms / 1000.0 * fs)
    is_still = speed < 5.0  # px/s
    pauses = 0
    count = 0
    for s in is_still:
        if s:
            count += 1
        else:
            if count >= min_samples:
                pauses += 1
            count = 0
    if count >= min_samples:
        pauses += 1
    return pauses


def extract_features_from_window(x, y, speed, accel, jerk, fs, raw_df_window=None):
    """Extract all features from a single time window."""
    features = {}

    # --- Velocity features ---
    features["velocity_mean"] = np.mean(speed)
    features["velocity_std"] = np.std(speed)
    features["velocity_max"] = np.max(speed)
    features["velocity_median"] = np.median(speed)

    # --- Acceleration features ---
    features["accel_mean"] = np.mean(accel)
    features["accel_std"] = np.std(accel)
    features["accel_max"] = np.max(accel)

    # --- Jerk features (movement smoothness) ---
    features["jerk_rms"] = np.sqrt(np.mean(jerk**2))
    features["jerk_std"] = np.std(jerk)
    features["jerk_max"] = np.max(jerk)

    # --- Path efficiency ---
    features["path_efficiency"] = compute_path_efficiency(x, y)

    # --- Pause features ---
    duration_sec = len(speed) / fs
    n_pauses = count_pauses(speed, fs)
    features["pause_count"] = n_pauses
    features["pause_frequency"] = n_pauses / (duration_sec / 60) if duration_sec > 0 else 0

    # --- FFT features on idle segments ---
    idle_mask = speed < IDLE_VELOCITY_THRESHOLD
    idle_x = x[idle_mask] if np.sum(idle_mask) > 0 else x
    idle_y = y[idle_mask] if np.sum(idle_mask) > 0 else y

    # FFT on x-axis idle signal
    tremor_ratio_x, spec_ent_x, _, _ = compute_fft_features(idle_x, fs)
    tremor_ratio_y, spec_ent_y, _, _ = compute_fft_features(idle_y, fs)

    features["tremor_energy_ratio"] = (tremor_ratio_x + tremor_ratio_y) / 2
    features["spectral_entropy"] = (spec_ent_x + spec_ent_y) / 2

    # --- Movement variability ---
    if len(x) > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.arctan2(dy, dx)
        features["direction_change_std"] = np.std(np.diff(angles)) if len(angles) > 1 else 0
    else:
        features["direction_change_std"] = 0

    # --- Click features (from raw data if available) ---
    if raw_df_window is not None and len(raw_df_window) > 0:
        clicks = raw_df_window[raw_df_window["event_type"] == "click_down"]
        if len(clicks) > 1:
            click_intervals = np.diff(clicks["timestamp_ms"].values)
            features["click_interval_mean"] = np.mean(click_intervals)
            features["click_interval_cv"] = np.std(click_intervals) / np.mean(click_intervals) if np.mean(click_intervals) > 0 else 0
        else:
            features["click_interval_mean"] = 0
            features["click_interval_cv"] = 0
        features["click_rate"] = len(clicks) / (duration_sec / 60) if duration_sec > 0 else 0
    else:
        features["click_interval_mean"] = 0
        features["click_interval_cv"] = 0
        features["click_rate"] = 0

    return features


def process_session(session_info, target_fs=200):
    """
    Full processing pipeline for one session:
    Load → Filter move events → Resample → Filter → Derivatives → Windowed features
    """
    print(f"  Processing: {session_info['participant']}/{session_info['session_id']}")

    # Load data
    df = load_session(session_info["csv_path"])
    if len(df) < 1000:
        print(f"    ⚠️ Too few events ({len(df)}), skipping")
        return []

    # Assign labels
    df = assign_time_labels(df)

    # Get move events only for signal processing
    df_move = df[df["event_type"] == "move"].copy()
    if len(df_move) < 500:
        print(f"    ⚠️ Too few move events ({len(df_move)}), skipping")
        return []

    # Estimate original sampling rate
    orig_rate = estimate_sampling_rate(df_move["timestamp_ms"].values)
    print(f"    Original rate: ~{orig_rate:.0f} Hz, Events: {len(df):,}")

    # Resample to uniform grid
    df_uniform = resample_to_uniform(df_move, target_rate_hz=target_fs)
    if df_uniform is None or len(df_uniform) < target_fs * 10:
        print(f"    ⚠️ Could not resample, skipping")
        return []

    # Apply Butterworth filter
    x_filt = apply_butterworth_filter(df_uniform["x"].values, BUTTERWORTH_CUTOFF_HZ, target_fs)
    y_filt = apply_butterworth_filter(df_uniform["y"].values, BUTTERWORTH_CUTOFF_HZ, target_fs)

    # Compute derivatives
    speed, accel, jerk, vx, vy = compute_derivatives(x_filt, y_filt, target_fs)

    # Assign labels to uniform data (based on time)
    t0 = df["timestamp_ms"].iloc[0]
    elapsed_ms = df_uniform["timestamp_ms"].values - t0
    conditions = [
        elapsed_ms < ALERT_MINUTES * 60 * 1000,
        (elapsed_ms >= ALERT_MINUTES * 60 * 1000) & (elapsed_ms < FATIGUED_MINUTES * 60 * 1000),
        elapsed_ms >= FATIGUED_MINUTES * 60 * 1000,
    ]
    labels_uniform = np.select(conditions, ["Alert", "Transition", "Fatigued"], default="Transition")

    # Sliding window feature extraction
    window_samples = int(WINDOW_SIZE_SEC * target_fs)
    step_samples = int(WINDOW_STEP_SEC * target_fs)
    n_samples = len(x_filt)

    all_features = []
    for start in range(0, n_samples - window_samples, step_samples):
        end = start + window_samples
        win_labels = labels_uniform[start:end]

        # Skip windows that span multiple labels
        unique_labels = np.unique(win_labels)
        if len(unique_labels) > 1:
            continue

        label = unique_labels[0]
        # Skip "Transition" for training (ambiguous)
        if label == "Transition":
            continue

        # Extract features
        win_x = x_filt[start:end]
        win_y = y_filt[start:end]
        win_speed = speed[start:end]
        win_accel = accel[start:end]
        win_jerk = jerk[start:end]

        # Get corresponding raw events for click analysis
        win_t_start = df_uniform["timestamp_ms"].iloc[start]
        win_t_end = df_uniform["timestamp_ms"].iloc[end - 1]
        raw_window = df[
            (df["timestamp_ms"] >= win_t_start) &
            (df["timestamp_ms"] <= win_t_end)
        ]

        feat = extract_features_from_window(
            win_x, win_y, win_speed, win_accel, win_jerk,
            target_fs, raw_df_window=raw_window
        )
        feat["label"] = label
        feat["participant"] = session_info["participant"]
        feat["session_id"] = session_info["session_id"]
        feat["window_start_ms"] = win_t_start
        all_features.append(feat)

    print(f"    ✓ Extracted {len(all_features)} windows "
          f"(Alert: {sum(1 for f in all_features if f['label']=='Alert')}, "
          f"Fatigued: {sum(1 for f in all_features if f['label']=='Fatigued')})")

    return all_features


# ============================================================
# Step 4: ML Training & Evaluation
# ============================================================

FEATURE_COLS = [
    "velocity_mean", "velocity_std", "velocity_max", "velocity_median",
    "accel_mean", "accel_std", "accel_max",
    "jerk_rms", "jerk_std", "jerk_max",
    "path_efficiency",
    "pause_count", "pause_frequency",
    "tremor_energy_ratio", "spectral_entropy",
    "direction_change_std",
    "click_interval_mean", "click_interval_cv", "click_rate",
]


def train_and_evaluate(features_df):
    """Train Random Forest and evaluate with cross-validation."""
    print("\n" + "=" * 60)
    print("  🤖 ML Training & Evaluation")
    print("=" * 60)

    X = features_df[FEATURE_COLS].values
    y = (features_df["label"] == "Fatigued").astype(int).values  # 0=Alert, 1=Fatigued
    groups = features_df["participant"].values

    print(f"\n  Dataset: {len(X)} samples")
    print(f"  Alert: {np.sum(y==0)}, Fatigued: {np.sum(y==1)}")
    print(f"  Participants: {features_df['participant'].nunique()}")
    print(f"  Features: {len(FEATURE_COLS)}")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Model 1: Random Forest (Primary) ----
    print("\n  --- Random Forest ---")
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=42, n_jobs=-1, class_weight="balanced"
        ))
    ])

    # Cross-validation
    # Determine max feasible splits based on minority class size
    min_class_count = min(np.sum(y==0), np.sum(y==1))
    n_unique_groups = len(np.unique(groups))
    
    if min_class_count < 2:
        print("  ⚠️ Not enough samples in one class for CV. Training on full data.")
        rf.fit(X, y)
        # Fake cv_results for downstream compatibility
        y_pred = rf.predict(X)
        cv_results = {
            "test_accuracy": np.array([accuracy_score(y, y_pred)]),
            "test_precision": np.array([0]),
            "test_recall": np.array([0]),
            "test_f1": np.array([0]),
            "test_roc_auc": np.array([0]),
            "estimator": [rf],
        }
        print(f"  Training Accuracy: {cv_results['test_accuracy'][0]:.3f}")
    else:
        n_cv_splits = min(5, min_class_count)
        
        if n_unique_groups >= 3 and min_class_count >= n_unique_groups:
            cv = StratifiedGroupKFold(n_splits=min(n_cv_splits, n_unique_groups))
            cv_results = cross_validate(rf, X, y, cv=cv, groups=groups,
                                        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                        return_estimator=True)
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
            cv_results = cross_validate(rf, X, y, cv=cv,
                                        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                        return_estimator=True)

        print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
        print(f"  Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}")
        print(f"  Recall:    {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}")
        print(f"  F1 Score:  {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
        print(f"  AUC-ROC:   {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")

    # Train final model on all data
    rf.fit(X, y)

    # Use simple StratifiedKFold for baselines (more robust with small data)
    from sklearn.model_selection import StratifiedKFold
    min_class_n = min(np.sum(y==0), np.sum(y==1))
    if min_class_n < 2:
        print("  ⚠️ Too few samples in minority class for baseline CV, skipping baselines.")
        return rf, cv_results
    
    n_baseline_splits = max(2, min(5, min_class_n))
    simple_cv = StratifiedKFold(n_splits=n_baseline_splits, shuffle=True, random_state=42)

    # ---- Model 2: Logistic Regression (Baseline) ----
    print("\n  --- Logistic Regression (Baseline) ---")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    try:
        lr_results = cross_validate(lr, X, y, cv=simple_cv,
                                    scoring=["accuracy", "f1", "roc_auc"])
        print(f"  Accuracy:  {lr_results['test_accuracy'].mean():.3f} ± {lr_results['test_accuracy'].std():.3f}")
        print(f"  F1 Score:  {lr_results['test_f1'].mean():.3f} ± {lr_results['test_f1'].std():.3f}")
        print(f"  AUC-ROC:   {lr_results['test_roc_auc'].mean():.3f} ± {lr_results['test_roc_auc'].std():.3f}")
    except Exception as e:
        print(f"  ⚠️ Logistic Regression CV failed: {e}")
        lr_results = {"test_accuracy": np.array([0]), "test_f1": np.array([0]), "test_roc_auc": np.array([0])}

    # ---- Model 3: SVM (Baseline) ----
    print("\n  --- SVM (Baseline) ---")
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
    ])
    try:
        svm_results = cross_validate(svm, X, y, cv=simple_cv,
                                     scoring=["accuracy", "f1", "roc_auc"])
        print(f"  Accuracy:  {svm_results['test_accuracy'].mean():.3f} ± {svm_results['test_accuracy'].std():.3f}")
        print(f"  F1 Score:  {svm_results['test_f1'].mean():.3f} ± {svm_results['test_f1'].std():.3f}")
        print(f"  AUC-ROC:   {svm_results['test_roc_auc'].mean():.3f} ± {svm_results['test_roc_auc'].std():.3f}")
    except Exception as e:
        print(f"  ⚠️ SVM CV failed: {e}")
        svm_results = {"test_accuracy": np.array([0]), "test_f1": np.array([0]), "test_roc_auc": np.array([0])}

    return rf, cv_results


# ============================================================
# Step 5: Visualization & Reports
# ============================================================

def plot_feature_importance(model, output_dir):
    """Plot Random Forest feature importance."""
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E8651A" if i < 5 else "#666666" for i in range(len(FEATURE_COLS))]
    sorted_colors = [colors[indices[i]] for i in range(len(indices))]

    ax.barh(range(len(FEATURE_COLS)),
            importances[indices],
            color=sorted_colors)
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest - Feature Importance Ranking")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()
    print(f"  📊 Saved: {output_dir / 'feature_importance.png'}")


def plot_confusion_matrix(model, X, y, output_dir):
    """Plot confusion matrix."""
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Alert", "Fatigued"])
    disp.plot(ax=ax, cmap="Oranges")
    ax.set_title("Confusion Matrix (Full Dataset)")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  📊 Saved: {output_dir / 'confusion_matrix.png'}")


def plot_feature_distributions(features_df, output_dir):
    """Plot distributions of key features by label."""
    key_features = [
        "velocity_mean", "jerk_rms", "tremor_energy_ratio",
        "spectral_entropy", "path_efficiency", "pause_frequency"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, feat in enumerate(key_features):
        ax = axes[i // 3][i % 3]
        for label, color in [("Alert", "#333333"), ("Fatigued", "#E8651A")]:
            data = features_df[features_df["label"] == label][feat]
            ax.hist(data, bins=30, alpha=0.6, label=label, color=color, density=True)
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=9)
    plt.suptitle("Feature Distributions: Alert vs Fatigued", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=150)
    plt.close()
    print(f"  📊 Saved: {output_dir / 'feature_distributions.png'}")


def plot_signal_comparison(session_info, output_dir, target_fs=200):
    """Plot before/after filter comparison and FFT for a single session."""
    df = load_session(session_info["csv_path"])
    df_move = df[df["event_type"] == "move"].copy()

    # Take a 10-second sample from the middle
    df_uniform = resample_to_uniform(df_move, target_rate_hz=target_fs)
    if df_uniform is None:
        return

    mid = len(df_uniform) // 2
    sample_len = target_fs * 10  # 10 seconds
    start = max(0, mid - sample_len // 2)
    end = min(len(df_uniform), start + sample_len)

    x_raw = df_uniform["x"].values[start:end]
    y_raw = df_uniform["y"].values[start:end]
    t = np.arange(len(x_raw)) / target_fs

    # Filter
    x_filt = apply_butterworth_filter(x_raw, BUTTERWORTH_CUTOFF_HZ, target_fs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Raw vs Filtered X
    axes[0][0].plot(t, x_raw, alpha=0.5, color="#999999", linewidth=0.5, label="Raw")
    axes[0][0].plot(t, x_filt, color="#E8651A", linewidth=1.0, label="Filtered")
    axes[0][0].set_title("X Position: Raw vs Butterworth Filtered")
    axes[0][0].set_xlabel("Time (s)")
    axes[0][0].set_ylabel("X (px)")
    axes[0][0].legend()

    # Compute speed
    speed_raw = np.sqrt(np.gradient(x_raw, 1/target_fs)**2 + np.gradient(y_raw, 1/target_fs)**2)
    y_filt = apply_butterworth_filter(y_raw, BUTTERWORTH_CUTOFF_HZ, target_fs)
    speed_filt = np.sqrt(np.gradient(x_filt, 1/target_fs)**2 + np.gradient(y_filt, 1/target_fs)**2)

    axes[0][1].plot(t, speed_raw, alpha=0.4, color="#999999", linewidth=0.5, label="Raw")
    axes[0][1].plot(t, speed_filt, color="#E8651A", linewidth=0.8, label="Filtered")
    axes[0][1].set_title("Speed: Raw vs Filtered")
    axes[0][1].set_xlabel("Time (s)")
    axes[0][1].set_ylabel("Speed (px/s)")
    axes[0][1].legend()

    # FFT of raw x
    n = len(x_raw)
    yf = np.abs(fft(x_raw * np.hanning(n)))[:n//2]
    xf = fftfreq(n, 1.0/target_fs)[:n//2]
    axes[1][0].plot(xf, yf**2, color="#333333", linewidth=0.8)
    axes[1][0].axvspan(TREMOR_BAND_LOW, TREMOR_BAND_HIGH, alpha=0.2, color="#E8651A", label="Tremor band (8-12 Hz)")
    axes[1][0].set_title("FFT Power Spectrum")
    axes[1][0].set_xlabel("Frequency (Hz)")
    axes[1][0].set_ylabel("Power")
    axes[1][0].set_xlim(0, 50)
    axes[1][0].legend()

    # Jerk profile
    _, _, jerk, _, _ = compute_derivatives(x_filt, y_filt, target_fs)
    axes[1][1].plot(t, jerk, color="#E8651A", linewidth=0.5)
    axes[1][1].set_title("Jerk Profile (Movement Smoothness)")
    axes[1][1].set_xlabel("Time (s)")
    axes[1][1].set_ylabel("Jerk (px/s³)")

    plt.suptitle(f"Signal Analysis: {session_info['participant']}/{session_info['session_id']}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"signal_{session_info['participant']}_{session_info['session_id']}.png"
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()
    print(f"  📊 Saved: {output_dir / fname}")


def plot_model_comparison(cv_rf, cv_lr, cv_svm, output_dir):
    """Bar chart comparing the 3 models."""
    models = ["Random Forest", "Logistic Reg.", "SVM"]
    metrics = {
        "Accuracy": [cv_rf["test_accuracy"].mean(), cv_lr["test_accuracy"].mean(), cv_svm["test_accuracy"].mean()],
        "F1": [cv_rf["test_f1"].mean(), cv_lr["test_f1"].mean(), cv_svm["test_f1"].mean()],
        "AUC-ROC": [cv_rf["test_roc_auc"].mean(), cv_lr["test_roc_auc"].mean(), cv_svm["test_roc_auc"].mean()],
    }

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#E8651A", "#F28C38", "#333333"]
    for i, (metric, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric, color=colors[i])

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")

    for i, (metric, values) in enumerate(metrics.items()):
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()
    print(f"  📊 Saved: {output_dir / 'model_comparison.png'}")


def generate_report(features_df, cv_results, output_dir):
    """Generate a summary text report."""
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PsychoMouse Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total windows:     {len(features_df)}\n")
        f.write(f"Alert windows:     {len(features_df[features_df['label']=='Alert'])}\n")
        f.write(f"Fatigued windows:  {len(features_df[features_df['label']=='Fatigued'])}\n")
        f.write(f"Participants:      {features_df['participant'].nunique()}\n")
        f.write(f"Sessions:          {features_df.groupby(['participant','session_id']).ngroups}\n\n")

        f.write("RANDOM FOREST RESULTS (Cross-Validation)\n")
        f.write("-" * 40 + "\n")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            key = f"test_{metric}"
            f.write(f"{metric.upper():12s}: {cv_results[key].mean():.3f} ± {cv_results[key].std():.3f}\n")

        f.write(f"\nFEATURE IMPORTANCE (Top 10)\n")
        f.write("-" * 40 + "\n")
        best_model = cv_results["estimator"][0]
        clf = best_model.named_steps["clf"]
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(FEATURE_COLS))):
            f.write(f"  {i+1:2d}. {FEATURE_COLS[indices[i]]:25s} {importances[indices[i]]:.4f}\n")

    print(f"  📄 Saved: {report_path}")


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline():
    """Run the complete analysis pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  🧠 PsychoMouse Analysis Pipeline")
    print("=" * 60)

    # Find all sessions
    sessions = find_all_sessions()
    print(f"\n  Found {len(sessions)} sessions from {len(set(s['participant'] for s in sessions))} participants:")
    for s in sessions:
        print(f"    - {s['participant']}/{s['session_id']}")

    if not sessions:
        print("  ❌ No data found in 'data/' directory!")
        return

    # Process all sessions
    print("\n" + "-" * 60)
    print("  Step 1: Signal Processing & Feature Extraction")
    print("-" * 60)

    all_features = []
    for session in sessions:
        features = process_session(session)
        all_features.extend(features)

    if not all_features:
        print("  ❌ No features extracted! Check your data.")
        return

    features_df = pd.DataFrame(all_features)
    features_csv = OUTPUT_DIR / "features.csv"
    features_df.to_csv(features_csv, index=False)
    print(f"\n  ✓ Total: {len(features_df)} feature windows saved to {features_csv}")

    # Train & Evaluate
    print("\n" + "-" * 60)
    print("  Step 2: ML Training & Evaluation")
    print("-" * 60)

    rf_model, rf_cv = train_and_evaluate(features_df)

    # Also get LR and SVM results for comparison plot
    X = np.nan_to_num(features_df[FEATURE_COLS].values)
    y = (features_df["label"] == "Fatigued").astype(int).values
    groups = features_df["participant"].values

    n_splits = min(5, len(np.unique(groups)))
    from sklearn.model_selection import StratifiedKFold
    min_class_n = min(np.sum(y==0), np.sum(y==1))
    n_baseline_splits = max(2, min(5, min_class_n))
    simple_cv = StratifiedKFold(n_splits=n_baseline_splits, shuffle=True, random_state=42)

    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    try:
        lr_cv = cross_validate(lr, X, y, cv=simple_cv, scoring=["accuracy", "f1", "roc_auc"])
    except Exception:
        lr_cv = {"test_accuracy": np.array([0]), "test_f1": np.array([0]), "test_roc_auc": np.array([0])}

    svm = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))])
    try:
        svm_cv = cross_validate(svm, X, y, cv=simple_cv, scoring=["accuracy", "f1", "roc_auc"])
    except Exception:
        svm_cv = {"test_accuracy": np.array([0]), "test_f1": np.array([0]), "test_roc_auc": np.array([0])}

    # Save model
    model_path = MODEL_DIR / "rf_model.joblib"
    joblib.dump(rf_model, model_path)
    print(f"\n  💾 Model saved: {model_path}")

    # Generate visualizations
    print("\n" + "-" * 60)
    print("  Step 3: Generating Visualizations")
    print("-" * 60)

    plot_feature_importance(rf_model, OUTPUT_DIR)
    plot_confusion_matrix(rf_model, X, y, OUTPUT_DIR)
    plot_feature_distributions(features_df, OUTPUT_DIR)
    plot_model_comparison(rf_cv, lr_cv, svm_cv, OUTPUT_DIR)

    # Signal plots for first 2 sessions
    for session in sessions[:2]:
        try:
            plot_signal_comparison(session, OUTPUT_DIR)
        except Exception as e:
            print(f"    ⚠️ Could not plot signal for {session['participant']}: {e}")

    # Generate report
    generate_report(features_df, rf_cv, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Results directory: {OUTPUT_DIR}/")
    print(f"  Key files:")
    print(f"    📄 features.csv          - Extracted features")
    print(f"    📄 analysis_report.txt   - Summary report")
    print(f"    📊 feature_importance.png")
    print(f"    📊 confusion_matrix.png")
    print(f"    📊 feature_distributions.png")
    print(f"    📊 model_comparison.png")
    print(f"    📊 signal_*.png          - Signal visualizations")
    print(f"    💾 models/rf_model.joblib - Trained model")
    print()


def label_all_sessions():
    """Label all sessions in the data directory."""
    sessions = find_all_sessions()
    print(f"Found {len(sessions)} sessions to label.\n")
    for s in sessions:
        print(f"Labeling: {s['participant']}/{s['session_id']}")
        df = load_session(s["csv_path"])
        df = assign_time_labels(df)

        output_path = s["session_dir"] / "mouse_events_labeled.csv"
        df.to_csv(output_path, index=False)

        counts = df["label"].value_counts()
        for label in ["Alert", "Transition", "Fatigued"]:
            n = counts.get(label, 0)
            print(f"  {label}: {n:,}")
        print(f"  Saved: {output_path}\n")


def visualize_session(session_dir_path):
    """Visualize a single session."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    session_dir = Path(session_dir_path)
    csv_path = session_dir / "mouse_events.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    session_info = {
        "participant": session_dir.parent.name,
        "session_id": session_dir.name,
        "session_dir": session_dir,
        "csv_path": csv_path,
    }
    plot_signal_comparison(session_info, OUTPUT_DIR)
    print("Done!")


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PsychoMouse Analysis Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # run: full pipeline
    subparsers.add_parser("run", help="Run full analysis pipeline")

    # label-all: label all sessions
    subparsers.add_parser("label-all", help="Assign time-based labels to all sessions")

    # visualize: single session visualization
    vis_parser = subparsers.add_parser("visualize", help="Visualize a single session")
    vis_parser.add_argument("session_dir", help="Path to session directory")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline()
    elif args.command == "label-all":
        label_all_sessions()
    elif args.command == "visualize":
        visualize_session(args.session_dir)
    else:
        parser.print_help()
        print("\n  Quick start: python analyze.py run\n")


if __name__ == "__main__":
    main()