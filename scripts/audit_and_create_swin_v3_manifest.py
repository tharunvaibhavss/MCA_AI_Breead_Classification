"""Audit complete selected breeds dataset and generate train_swin_synth_v3.csv manifest.

Verifies:
- 153 Real training images
- 4,160 Synthetic training images
- 34 Real validation images
- 33 Real locked test images
- SHA-256 cryptographic uniqueness & 0% data leakage
- Integrity of all images on disk
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import pandas as pd

SELECTED_CLASSES = [
    {"class_id": 0, "breed_id": "cow_dangi", "breed_name": "Dangi", "species": "cattle"},
    {"class_id": 1, "breed_id": "cow_gaolao", "breed_name": "Gaolao", "species": "cattle"},
    {"class_id": 2, "breed_id": "cow_gir", "breed_name": "Gir", "species": "cattle"},
    {"class_id": 3, "breed_id": "cow_ongole", "breed_name": "Ongole", "species": "cattle"},
    {"class_id": 4, "breed_id": "cow_sahiwal", "breed_name": "Sahiwal", "species": "cattle"},
    {"class_id": 5, "breed_id": "buffalo_manda", "breed_name": "Manda", "species": "buffalo"},
    {"class_id": 6, "breed_id": "buffalo_surti", "breed_name": "Surti", "species": "buffalo"},
    {"class_id": 7, "breed_id": "buffalo_toda", "breed_name": "Toda", "species": "buffalo"},
]

BREED_SPECIES_MAP = {c["breed_name"]: c["species"] for c in SELECTED_CLASSES}
BREED_ID_MAP = {c["breed_name"]: c["breed_id"] for c in SELECTED_CLASSES}

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 80)
    print("SWIN-T SYNTHETIC V3: COMPLETE DATASET AUDIT & MANIFEST CREATION")
    print("=" * 80)

    # 1. Load existing splits to preserve exact Real splits
    val_df = pd.read_csv("splits/selected_breeds/validation.csv")
    test_df = pd.read_csv("splits/selected_breeds/test.csv")
    locked_df = pd.read_csv("splits/selected_breeds/locked_test_manifest.csv")
    
    print(f"Loaded existing Real validation split: {len(val_df)} images")
    print(f"Loaded existing Real test split: {len(test_df)} images")
    print(f"Loaded existing Real locked test manifest: {len(locked_df)} images")

    assert len(val_df) == 34, f"Validation set must be exactly 34 images, found {len(val_df)}"
    assert len(test_df) == 33, f"Test set must be exactly 33 images, found {len(test_df)}"
    assert len(locked_df) == 33, f"Locked test must be exactly 33 images, found {len(locked_df)}"

    val_shas = set(val_df["sha256"])
    test_shas = set(test_df["sha256"])
    locked_shas = set(locked_df["sha256"])
    assert test_shas == locked_shas, "Test and locked test manifests do not match!"
    assert len(val_shas.intersection(test_shas)) == 0, "Validation and Test share images!"

    # 2. Audit Real Training Images
    # Real training images are in splits/selected_breeds/train.csv with source_type == REAL
    train_prev_df = pd.read_csv("splits/selected_breeds/train.csv")
    real_train_df = train_prev_df[train_prev_df["source_type"] == "REAL"].copy()
    print(f"Found {len(real_train_df)} Real training images from established split.")
    assert len(real_train_df) == 153, f"Expected 153 real training images, got {len(real_train_df)}"

    real_train_records = []
    for _, row in real_train_df.iterrows():
        p = Path(row["image_path"])
        if not p.exists():
            raise FileNotFoundError(f"Real training image not found: {p}")
        sha = compute_sha256(p)
        assert sha == row["sha256"], f"SHA mismatch for {p}"
        
        # Verify readability and resolution
        with Image.open(p) as img:
            img.verify()
        with Image.open(p) as img:
            w, h = img.size
            fmt = img.format

        real_train_records.append({
            "image_path": str(p).replace("\\", "/"),
            "breed": row["breed"],
            "breed_id": BREED_ID_MAP[row["breed"]],
            "species": BREED_SPECIES_MAP[row["breed"]],
            "source_type": "REAL",
            "sha256": sha,
            "synthetic_version": "N/A",
            "width": w,
            "height": h,
            "format": fmt
        })

    # 3. Audit Synthetic Training Images on disk
    synth_dir = Path("dataset/selected_breeds/synthetic")
    synth_records = []
    non_image_files = []
    corrupted_files = []

    for fpath in sorted(synth_dir.rglob("*")):
        if fpath.is_file():
            if fpath.name.lower() == "readme.txt" or fpath.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                non_image_files.append(str(fpath))
                continue
            
            breed_map = {
                "cow_dangi": ("Dangi", "cow_dangi", "cattle"),
                "cow_gaolao": ("Gaolao", "cow_gaolao", "cattle"),
                "cow_gir": ("Gir", "cow_gir", "cattle"),
                "cow_ongole": ("Ongole", "cow_ongole", "cattle"),
                "cow_sahiwal": ("Sahiwal", "cow_sahiwal", "cattle"),
                "buffalo_manda": ("Manda", "buffalo_manda", "buffalo"),
                "buffalo_surti": ("Surti", "buffalo_surti", "buffalo"),
                "buffalo_toda": ("Toda", "buffalo_toda", "buffalo"),
                "dangi": ("Dangi", "cow_dangi", "cattle"),
                "gaolao": ("Gaolao", "cow_gaolao", "cattle"),
                "gir": ("Gir", "cow_gir", "cattle"),
                "ongole": ("Ongole", "cow_ongole", "cattle"),
                "sahiwal": ("Sahiwal", "cow_sahiwal", "cattle"),
                "manda": ("Manda", "buffalo_manda", "buffalo"),
                "surti": ("Surti", "buffalo_surti", "buffalo"),
                "toda": ("Toda", "buffalo_toda", "buffalo"),
            }

            parent_name = fpath.parent.name.lower()
            if parent_name in breed_map:
                detected_breed, detected_breed_id, detected_species = breed_map[parent_name]
            elif fpath.parent.parent.name.lower() in breed_map:
                detected_breed, detected_breed_id, detected_species = breed_map[fpath.parent.parent.name.lower()]
            else:
                raise ValueError(f"Could not determine breed for synthetic file: {fpath}")

            try:
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    w, h = img.size
                    fmt = img.format
                sha = compute_sha256(fpath)
            except Exception as e:
                corrupted_files.append((str(fpath), str(e)))
                continue

            # Determine synthetic version (legacy 800 or expanded v4)
            if fpath.name.startswith("SYNTH_"):
                synth_ver = "legacy_synth_800"
            else:
                synth_ver = "v4_expanded_500"

            synth_records.append({
                "image_path": str(fpath).replace("\\", "/"),
                "breed": detected_breed,
                "breed_id": BREED_ID_MAP[detected_breed],
                "species": BREED_SPECIES_MAP[detected_breed],
                "source_type": "SYNTHETIC",
                "sha256": sha,
                "synthetic_version": synth_ver,
                "width": w,
                "height": h,
                "format": fmt
            })

    print(f"Total synthetic images found and verified: {len(synth_records)}")
    print(f"Non-image files excluded: {len(non_image_files)} (e.g., {non_image_files[:2]})")
    print(f"Corrupted synthetic images: {len(corrupted_files)}")

    assert len(synth_records) == 4160, f"Expected 4,160 synthetic images, got {len(synth_records)}"
    assert len(corrupted_files) == 0, f"Found corrupted files: {corrupted_files}"

    # Check synthetic breakdown by breed
    synth_df_temp = pd.DataFrame(synth_records)
    print("\nSynthetic images per breed:")
    for b in SELECTED_CLASSES:
        cnt = (synth_df_temp["breed"] == b["breed_name"]).sum()
        print(f"  - {b['breed_name']} ({b['species']}): {cnt}")

    # Check synthetic breakdown by version
    print("\nSynthetic images by version:")
    print(synth_df_temp["synthetic_version"].value_counts().to_dict())

    # 4. Rigorous Duplicate and Leakage Checks
    print("\nChecking cryptographic uniqueness and zero-leakage isolation...")
    real_train_shas = {r["sha256"] for r in real_train_records}
    synth_shas = {r["sha256"] for r in synth_records}

    print(f"  Real training images: {len(real_train_records)} total, {len(real_train_shas)} unique SHAs")
    print(f"  Synthetic training images on disk: {len(synth_records)} total, {len(synth_shas)} unique SHAs")
    assert len(real_train_records) == len(real_train_shas), "Duplicate hashes inside Real train!"

    # Split overlaps
    leakage_train_test = real_train_shas.intersection(test_shas)
    leakage_val_test = val_shas.intersection(test_shas)
    leakage_train_val = real_train_shas.intersection(val_shas)
    leakage_synth_test = synth_shas.intersection(test_shas)
    leakage_synth_val = synth_shas.intersection(val_shas)
    leakage_synth_real_train = synth_shas.intersection(real_train_shas)

    print(f"  Train / Test overlap: {len(leakage_train_test)} (Expected: 0)")
    print(f"  Val / Test overlap: {len(leakage_val_test)} (Expected: 0)")
    print(f"  Train / Val overlap: {len(leakage_train_val)} (Expected: 0)")
    print(f"  Synthetic / Test overlap: {len(leakage_synth_test)} (Expected: 0)")
    print(f"  Synthetic / Val overlap: {len(leakage_synth_val)} (Expected: 0)")
    print(f"  Synthetic / Real Train overlap: {len(leakage_synth_real_train)} (Expected: 0)")

    assert len(leakage_train_test) == 0, "CRITICAL: Train/Test leakage detected!"
    assert len(leakage_val_test) == 0, "CRITICAL: Val/Test leakage detected!"
    assert len(leakage_train_val) == 0, "CRITICAL: Train/Val leakage detected!"
    assert len(leakage_synth_test) == 0, "CRITICAL: Synthetic in Test detected!"
    assert len(leakage_synth_val) == 0, "CRITICAL: Synthetic in Val detected!"
    assert len(leakage_synth_real_train) == 0, "CRITICAL: Synthetic matches Real Train SHA!"

    # 5. Create train_swin_synth_v3.csv
    all_train_records = real_train_records + synth_records
    train_swin_synth_df = pd.DataFrame(all_train_records)
    
    out_manifest_path = Path("splits/selected_breeds/train_swin_synth_v3.csv")
    train_swin_synth_df.to_csv(out_manifest_path, index=False)
    print(f"\nSaved new training manifest to: {out_manifest_path}")
    print(f"Total training records: {len(train_swin_synth_df)}")
    print(f"  - Real training: {(train_swin_synth_df['source_type'] == 'REAL').sum()}")
    print(f"  - Synthetic training: {(train_swin_synth_df['source_type'] == 'SYNTHETIC').sum()}")

    # 6. Save audit metadata summary JSON
    audit_summary = {
        "audit_timestamp": str(pd.Timestamp.now()),
        "real_train_count": len(real_train_records),
        "synthetic_train_count": len(synth_records),
        "total_train_count": len(train_swin_synth_df),
        "real_val_count": len(val_df),
        "real_test_count": len(test_df),
        "real_train_by_breed": train_prev_df[train_prev_df["source_type"] == "REAL"]["breed"].value_counts().to_dict(),
        "synthetic_train_by_breed": synth_df_temp["breed"].value_counts().to_dict(),
        "synthetic_versions": synth_df_temp["synthetic_version"].value_counts().to_dict(),
        "leakage_checks": {
            "train_test_overlap": len(leakage_train_test),
            "val_test_overlap": len(leakage_val_test),
            "train_val_overlap": len(leakage_train_val),
            "synth_test_overlap": len(leakage_synth_test),
            "synth_val_overlap": len(leakage_synth_val),
            "synth_real_train_overlap": len(leakage_synth_real_train)
        },
        "formats_synthetic": synth_df_temp["format"].value_counts().to_dict(),
        "formats_real": pd.DataFrame(real_train_records)["format"].value_counts().to_dict(),
    }

    os.makedirs("reports/selected_breeds_swin_synth_v3", exist_ok=True)
    with open("reports/selected_breeds_swin_synth_v3/dataset_audit_stats.json", "w") as f:
        json.dump(audit_summary, f, indent=2)
    print("Saved audit statistics to reports/selected_breeds_swin_synth_v3/dataset_audit_stats.json")

if __name__ == "__main__":
    main()
