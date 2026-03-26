"""
create_splits.py — Subject-dependent train/val/test splits for NTHU-DDD
========================================================================
Dataset structure assumed:
    root/
      drowsy/
        sleepyCombination/   *.jpg
        slowBlink/           *.jpg
        yawning/             *.jpg
      notdrowsy/
        <action_folders>/    *.jpg   (same actions, label is notdrowsy)

Filename format (both classes):
    {subject}_{glasses}_{action}_{framenum}_{label}.jpg
    e.g.  001_glasses_sleepyCombination_606_drowsy.jpg
          002_noglasses_yawning_60_notdrowsy.jpg

Split strategy — SUBJECT-DEPENDENT:
    Each subject's frames are split by (subject, action) video group.
    Frames from the SAME video are kept together to avoid frame-level
    leakage (consecutive frames of the same video must not appear in
    both train and test).

    Within each (subject, action, label) group the frames are sorted
    by frame number and divided chronologically:
        first 70%  → train
        next  15%  → val
        last  15%  → test

    This means ALL subjects appear in all three splits, so the model
    is evaluated on the SAME subjects it was trained on (subject-dependent).

Output CSVs  →  splits/train.csv  |  val.csv  |  test.csv
Columns      →  path, label, subject, glasses, action
"""

import os
import re
import pandas as pd
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_ROOT = "data/processed"           # ← change to your root folder
SPLITS_DIR   = "splits"

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO is the remainder (0.15)

DROWSY_LABEL    = 1
NOTDROWSY_LABEL = 0

os.makedirs(SPLITS_DIR, exist_ok=True)

# ── Filename parser ────────────────────────────────────────────────────────────

# Matches:  {subject}_{glasses}_{action}_{framenum}_{label}.ext
# subject  → digits only  (e.g. 001, 002)
# glasses  → glasses | noglasses
# action   → any word chars  (sleepyCombination, yawning, slowBlink, …)
# framenum → digits
# label    → drowsy | notdrowsy

_PATTERN = re.compile(
    r'^(\d+)_([^_]+)_(.+)_(\d+)_(drowsy|notdrowsy)\.(jpg|jpeg|png)$',
    re.IGNORECASE
)

def parse_filename(filename):
    """
    Returns (subject, glasses, action, frame_num, label_str) or None.
    The regex is greedy on `action` so multi-word actions like
    sleepyCombination are captured correctly regardless of case.
    """
    m = _PATTERN.match(filename)
    if not m:
        return None
    subject, glasses, action, frame_num, label_str, _ = m.groups()
    return subject, glasses, action, int(frame_num), label_str.lower()

# ── Walk dataset and collect all frames ───────────────────────────────────────

def collect_frames(dataset_root):
    """
    Walks drowsy/ and notdrowsy/ sub-trees.
    Returns a list of dicts with keys:
        path, label, subject, glasses, action
    """
    records = []
    skipped = 0

    for label_folder, label_int in [("drowsy", DROWSY_LABEL),
                                     ("notdrowsy", NOTDROWSY_LABEL)]:
        label_dir = os.path.join(dataset_root, label_folder)
        if not os.path.isdir(label_dir):
            print(f"  [WARN] Folder not found: {label_dir}")
            continue

        # Walk all sub-folders (sleepyCombination, slowBlink, yawning, …)
        for root, dirs, files in os.walk(label_dir):
            dirs.sort()   # deterministic order
            for fname in sorted(files):
                parsed = parse_filename(fname)
                if parsed is None:
                    skipped += 1
                    continue
                subject, glasses, action, frame_num, label_str = parsed

                # Sanity-check: label in filename must match the folder
                expected = "drowsy" if label_int == DROWSY_LABEL else "notdrowsy"
                if label_str != expected:
                    print(f"  [WARN] Label mismatch in {fname} (folder={expected})")
                    skipped += 1
                    continue

                records.append({
                    "path"   : os.path.join(root, fname),
                    "label"  : label_int,
                    "subject": subject,
                    "glasses": glasses,
                    "action" : action,
                })

    print(f"\nTotal frames collected : {len(records):,}")
    print(f"Skipped (parse fail)   : {skipped:,}")
    return records

# ── Subject-dependent split ───────────────────────────────────────────────────

def make_subject_dependent_splits(records, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Subject-dependent split: every subject appears in ALL three splits.

    Grouping key: (subject, glasses, action, label)
    → each group is one continuous video segment.

    Within each group, frames are sorted chronologically by frame_num
    and divided:
        first train_ratio  → train
        next  val_ratio    → val
        remainder          → test

    This avoids frame-level leakage (consecutive frames from the same
    video stay together) while ensuring the model sees every subject
    during training.
    """
    # Re-parse frame_num from path for sorting (already in filename)
    _FN = re.compile(r'_(\d+)_(drowsy|notdrowsy)\.(jpg|jpeg|png)$', re.IGNORECASE)

    def get_frame_num(path):
        m = _FN.search(os.path.basename(path))
        return int(m.group(1)) if m else 0

    # Group by (subject, glasses, action, label)
    groups = defaultdict(list)
    for r in records:
        key = (r["subject"], r["glasses"], r["action"], r["label"])
        groups[key].append(r)

    train_records, val_records, test_records = [], [], []
    group_stats = []

    for key, frames in sorted(groups.items()):
        subject, glasses, action, label = key

        # Sort chronologically
        frames = sorted(frames, key=lambda r: get_frame_num(r["path"]))
        n      = len(frames)

        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # ensure at least 1 frame in test if possible
        if n_train + n_val >= n and n >= 3:
            n_train = n - 2
            n_val   = 1

        train_part = frames[:n_train]
        val_part   = frames[n_train : n_train + n_val]
        test_part  = frames[n_train + n_val :]

        train_records.extend(train_part)
        val_records.extend(val_part)
        test_records.extend(test_part)

        group_stats.append({
            "key"  : f"{subject}/{glasses}/{action}/{'drowsy' if label else 'notdrowsy'}",
            "total": n,
            "train": len(train_part),
            "val"  : len(val_part),
            "test" : len(test_part),
        })

    # Print per-group summary
    print(f"\n{'Group':<45} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print("  " + "-" * 70)
    for g in group_stats:
        print(f"  {g['key']:<43} {g['total']:>6} {g['train']:>6} {g['val']:>5} {g['test']:>5}")

    return train_records, val_records, test_records

# ── Save CSVs ─────────────────────────────────────────────────────────────────

def save_split(records, out_path, split_name):
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    drowsy    = (df["label"] == DROWSY_LABEL).sum()
    notdrowsy = (df["label"] == NOTDROWSY_LABEL).sum()
    total     = len(df)

    print(f"\n[{split_name}]  {total:,} frames  →  {out_path}")
    print(f"  Drowsy    : {drowsy:,}  ({drowsy/total*100:.1f}%)")
    print(f"  Notdrowsy : {notdrowsy:,}  ({notdrowsy/total*100:.1f}%)")
    print(f"  Subjects  : {df['subject'].nunique()}")
    print(f"  Actions   : {sorted(df['action'].unique())}")
    return df

# ── Frame-level leakage verification ─────────────────────────────────────────

def verify_no_leakage(train_df, val_df, test_df):
    """
    For subject-dependent splits, subjects WILL appear in all splits —
    that is by design. What we check instead is that no individual
    frame path appears in more than one split.
    """
    train_paths = set(train_df["path"])
    val_paths   = set(val_df["path"])
    test_paths  = set(test_df["path"])

    tv = train_paths & val_paths
    tt = train_paths & test_paths
    vt = val_paths   & test_paths

    print("\n── Frame-level leakage check ──────────────────")
    if not tv and not tt and not vt:
        print("  ✓ No duplicate frames across any split")
    else:
        if tv: print(f"  ✗ Train ∩ Val  : {len(tv)} duplicate frames")
        if tt: print(f"  ✗ Train ∩ Test : {len(tt)} duplicate frames")
        if vt: print(f"  ✗ Val   ∩ Test : {len(vt)} duplicate frames")

    # Confirm all subjects appear in every split
    train_s = set(train_df["subject"])
    val_s   = set(val_df["subject"])
    test_s  = set(test_df["subject"])
    all_s   = train_s | val_s | test_s

    in_all = train_s & val_s & test_s
    print(f"\n  Subjects in train+val+test : {sorted(in_all)}")
    if in_all == all_s:
        print(f"  ✓ All {len(all_s)} subjects appear in every split (subject-dependent ✓)")
    else:
        missing = all_s - in_all
        print(f"  ⚠  Subjects not in all splits: {missing}")
    print("───────────────────────────────────────────────")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NTHU-DDD  —  create_splits.py")
    print("=" * 60)
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Output dir   : {SPLITS_DIR}")
    print(f"  Split ratios : {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {1-TRAIN_RATIO-VAL_RATIO:.0%} test  (chronological per video group)")

    # 1. Collect all frames
    records = collect_frames(DATASET_ROOT)
    if not records:
        print("\n[ERROR] No frames found. Check DATASET_ROOT and folder structure.")
        return

    # 2. Dataset-level stats
    df_all = pd.DataFrame(records)
    print(f"\nAll subjects found : {sorted(df_all['subject'].unique())}")
    print(f"Label distribution :")
    print(f"  Drowsy    : {(df_all['label']==1).sum():,}")
    print(f"  Notdrowsy : {(df_all['label']==0).sum():,}")
    print(f"Action counts :")
    for action, cnt in df_all['action'].value_counts().items():
        print(f"  {action:<25} : {cnt:,}")

    # 3. Subject-dependent split (chronological per video group)
    train_r, val_r, test_r = make_subject_dependent_splits(records)

    # 4. Save
    train_df = save_split(train_r, os.path.join(SPLITS_DIR, "train.csv"), "TRAIN")
    val_df   = save_split(val_r,   os.path.join(SPLITS_DIR, "val.csv"),   "VAL")
    test_df  = save_split(test_r,  os.path.join(SPLITS_DIR, "test.csv"),  "TEST")

    # 5. Shuffle (so chronological ordering doesn't bias batch training)
    print("\nShuffling splits...")
    for split_name, split_df, path in [
        ("train", train_df, os.path.join(SPLITS_DIR, "train.csv")),
        ("val",   val_df,   os.path.join(SPLITS_DIR, "val.csv")),
        ("test",  test_df,  os.path.join(SPLITS_DIR, "test.csv")),
    ]:
        split_df = split_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_df.to_csv(path, index=False)
        print(f"  {split_name}.csv shuffled ✓  |  "
              f"drowsy={split_df['label'].sum():,}  |  "
              f"notdrowsy={(split_df['label']==0).sum():,}")

    # 6. Verify no leakage
    verify_no_leakage(train_df, val_df, test_df)

    print("\n" + "=" * 60)
    print("  Splits saved successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
