"""
create_sequences.py — Build sequence CSVs for CNN+LSTM training
================================================================
Reads frame-level split CSVs produced by create_splits.py and groups
consecutive frames into fixed-length sequences using a sliding window.

Colab folder layout (from screenshot):
    /content/data/
        processed/          ← raw frames
        splits/             ← train.csv, val.csv, test.csv  (create_splits output)
        create_sequence.py  ← this file
        train_cnn_lstm.py

Output:
    /content/data/seq_splits/train_seq.csv
    /content/data/seq_splits/val_seq.csv
    /content/data/seq_splits/test_seq.csv

Usage:
    python /content/data/create_sequence.py
"""

import os
import re
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

SPLITS_DIR = "/content/data/splits"
SEQ_DIR    = "/content/data/seq_splits"

os.makedirs(SEQ_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

SEQ_LENGTH = 16    # frames per sequence (~0.5 s at 30 fps)
STRIDE     = 8     # step between windows (50 % overlap)

# ── Frame-number extractor (mirrors create_splits.py) ─────────────────────────

_FRAME_RE = re.compile(
    r'_(\d+)_(drowsy|notdrowsy)\.(jpg|jpeg|png)$',
    re.IGNORECASE
)

def get_frame_num(path: str) -> int:
    """Extract numeric frame index from filename. Returns 0 on failure."""
    m = _FRAME_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 0

# ── Core builder ──────────────────────────────────────────────────────────────

def create_sequence_csv(csv_path: str, out_path: str, split_name: str) -> pd.DataFrame:
    """
    Build a sequence CSV from a frame-level split CSV.

    Grouping key: (subject, glasses, action, label) — same key used in
    create_splits.py, so sequences never straddle a video boundary.
    """
    df = pd.read_csv(csv_path)
    print(f"\n[{split_name}] Building sequences from {len(df):,} frames …")
    print(f"  Columns : {list(df.columns)}")

    required = {"path", "label", "subject", "glasses", "action"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{split_name}] CSV missing columns: {missing}\n"
            f"Re-run create_splits.py to regenerate the split CSVs."
        )

    df["frame_num"] = df["path"].apply(get_frame_num)

    group_key = ["subject", "glasses", "action", "label"]
    df_sorted = df.sort_values(group_key + ["frame_num"]).reset_index(drop=True)

    sequences    = []
    groups_seen  = 0
    groups_short = 0

    for group_id, group in df_sorted.groupby(group_key, sort=True):
        subject, glasses, action, label = group_id
        video_id = f"{subject}_{glasses}_{action}"

        group    = group.sort_values("frame_num").reset_index(drop=True)
        paths    = group["path"].tolist()
        labels   = group["label"].tolist()
        n_frames = len(paths)
        groups_seen += 1

        if n_frames < SEQ_LENGTH:
            groups_short += 1
            continue

        for start in range(0, n_frames - SEQ_LENGTH + 1, STRIDE):
            seq_paths  = paths[start : start + SEQ_LENGTH]
            seq_labels = labels[start : start + SEQ_LENGTH]
            seq_label  = int(np.round(np.mean(seq_labels)))

            sequences.append({
                "video_id"  : video_id,
                "label"     : seq_label,
                "frames"    : "|".join(seq_paths),
                "seq_length": SEQ_LENGTH,
                "subject"   : subject,
                "action"    : action,
            })

    if not sequences:
        raise RuntimeError(
            f"[{split_name}] No sequences created. "
            f"SEQ_LENGTH={SEQ_LENGTH} may be larger than most video groups. "
            f"Check that frame paths in the CSV actually exist."
        )

    seq_df = pd.DataFrame(sequences)
    seq_df.to_csv(out_path, index=False)

    drowsy    = (seq_df["label"] == 1).sum()
    notdrowsy = (seq_df["label"] == 0).sum()
    total     = len(seq_df)

    print(f"  Groups processed  : {groups_seen}  ({groups_short} skipped — too short)")
    print(f"  Sequences created : {total:,}")
    print(f"    Drowsy          : {drowsy:,}  ({drowsy/total*100:.1f}%)")
    print(f"    Notdrowsy       : {notdrowsy:,}  ({notdrowsy/total*100:.1f}%)")
    print(f"  Unique subjects   : {seq_df['subject'].nunique()}")
    print(f"  Unique actions    : {sorted(seq_df['action'].unique())}")
    print(f"  Saved → {out_path}")
    return seq_df


# ── Run ───────────────────────────────────────────────────────────────────────

train_seq = create_sequence_csv(
    os.path.join(SPLITS_DIR, "train.csv"),
    os.path.join(SEQ_DIR,    "train_seq.csv"),
    "TRAIN",
)
val_seq = create_sequence_csv(
    os.path.join(SPLITS_DIR, "val.csv"),
    os.path.join(SEQ_DIR,    "val_seq.csv"),
    "VAL",
)
test_seq = create_sequence_csv(
    os.path.join(SPLITS_DIR, "test.csv"),
    os.path.join(SEQ_DIR,    "test_seq.csv"),
    "TEST",
)

sep = "=" * 55
print(f"\n{sep}")
print("  Sequence CSVs ready")
print(sep)
print(f"  Train : {len(train_seq):,} sequences")
print(f"  Val   : {len(val_seq):,} sequences")
print(f"  Test  : {len(test_seq):,} sequences")
print(f"\n  SEQ_LENGTH : {SEQ_LENGTH} frames")
print(f"  STRIDE     : {STRIDE} frames")
print(sep)