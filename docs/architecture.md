# Архитектура head-align

## Постановка

Задача — **3D rigid alignment** головы на КТ: оценить остаточный поворот (и сдвиг детектора) и один раз пересэмплировать HU-объём в каноническую укладку.

Координаты пайплайна — **индексная система изотропного slab @ 4 mm** с `direction = identity` (не patient LPS). Это согласовано с обучением и экспортом.

## Пайплайн inference / API

```
NIfTI (HU)
  → resample isotropic 4 mm
  → brain window [0,1] + square Y/X
  → bottom slab Z=48 (192 mm от нижнего конца стека)
  → cls (FullScanClsNet) на resize 48×56×56
  → axial PCA detector:
        3 аксиальных среза вверх от центра slab (step 10 px)
        маска: все пиксели > thr 20/255 на window
        PCA median → shift XY + rotZ
  → center crop 48×56×56 для pose
  → pose 1-pass (FullScanPoseNet) → rotvec residual
  → composed rigid transform (detector + pose)
  → один HU resample slab (expand_to_fit=false)
  → NIfTI + affine_4x4 в meta
```

## Модули

| Модуль | Назначение |
|--------|------------|
| `head_align/volume.py` | isotropic CT, infer slab, sitk helpers |
| `head_align/mask_utils.py` | порог маски (thr=20/255) |
| `head_align/axial_detector.py` | multi-slice axial PCA |
| `head_align/preprocess.py` | slab, axial PCA, center crop |
| `head_align/rigid.py` | compose transforms, resample |
| `head_align/inference.py` | `infer_case_hybrid` |
| `head_align/export.py` | HU export + `pipeline_meta` |
| `head_align/model.py` | 3D CNN cls / pose |
| `head_align/augment.py` | synthetic misalignment для train |
| `app/main.py` | FastAPI `/health`, `/align` |

## Модели

- **Cls** — бинарный «есть голова / нет» на pre-detector объёме.
- **Pose** — предсказывает **rotvec misalignment** (rad); на inference применяется **инверсия** (`-rotvec`). Translation head обучен, но в production pose применяется **только rotation**.

Backbone: 3D CNN `FullScanBackbone` → adaptive pool → linear heads.

## Ограничения

- Выход API — **infer slab** (48 срезов @ 4 mm), не полный исходный FOV.
- Допуск по ТЗ: ±5° по осям; на golden median residual pose ≈ 3°.
- 5 кейсов golden отсекаются cls (торс без головы / не head CT).
