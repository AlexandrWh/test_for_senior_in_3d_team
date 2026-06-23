#!/usr/bin/env python3
"""Прогон TotalSegmentator по volumes/*.nii.gz -> masks/<fname>/<class>.nii.gz"""
import os, glob, traceback
from totalsegmentator.python_api import totalsegmentator

VOLUMES_DIR = "data/volumes"
MASKS_DIR   = "data/masks"
DEVICE      = "gpu"     # "gpu" | "cpu" | "gpu:1" | "mps"
TASK        = "total"   # 117 структур
FAST        = False     # True -> 3мм модель: быстрее/грубее
ROI_SUBSET  = None      # напр. ["brain"] — считать только мозг; None -> все классы
SKIP_DONE   = True      # пропускать уже обработанные

def stem(path):
    n = os.path.basename(path)
    return n[:-7] if n.endswith(".nii.gz") else os.path.splitext(n)[0]

def main():
    vols = sorted(glob.glob(os.path.join(VOLUMES_DIR, "*.nii.gz")))
    if not vols:
        print(f"Нет файлов в {VOLUMES_DIR}/*.nii.gz"); return
    print(f"Найдено томов: {len(vols)}")

    ok, failed = 0, []
    for i, vol in enumerate(vols, 1):
        fname = stem(vol)
        out_dir = os.path.join(MASKS_DIR, fname)
        if SKIP_DONE and os.path.isdir(out_dir) and os.listdir(out_dir):
            print(f"[{i}/{len(vols)}] {fname}: уже есть, пропускаю"); ok += 1; continue
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{i}/{len(vols)}] {fname}: сегментирую...")
        try:
            totalsegmentator(
                input=vol, output=out_dir,
                task=TASK, fast=FAST, roi_subset=ROI_SUBSET,
                device=DEVICE, ml=False,   # ml=False -> один файл на класс
                quiet=True,
            )
            ok += 1
        except Exception:
            print(f"  ОШИБКА на {fname}:"); traceback.print_exc(); failed.append(fname)

    print(f"\nГотово. Успешно: {ok}/{len(vols)}.")
    if failed:
        print("Упали:", ", ".join(failed))

if __name__ == "__main__":
    main()