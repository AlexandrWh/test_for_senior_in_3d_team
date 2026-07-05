# Модули Python

Описание каждого `.py` файла (кроме `__init__.py`).

---

## `paths.py`

Единый источник путей и констант проекта:

- `data/cq500_train/*`, `data/cq500_test/*`
- `SPACING_MM = 4.0` (infer), `APPLY_SPACING_MM = 1.0` (финальный NIfTI)
- пути к весам, pose dataset, align output, mask residual
- `data/cq500_test/align/previews/` — golden PNG (в git)
- гиперпараметры аугментации pose (`POSE_AUG_RANGE_PI`, `POSE_AUG_PER_CASE`)

---

## `models/`

### `z_slice_head_cls.py`

CNN для бинарной классификации одного аксиального среза 56×56: «есть голова / нет».  
Метод `infer_head_z_span()` — по стеку срезов находит непрерывный Z-span головы (порог, pad, min slices).

### `pre_aligner.py`

**PreAligner** — первая стадия пайплайна:

- `prepare_volume()` — NIfTI → isotropic spacing → square crop → brain window
- `predict_params()` — Z-span (классификатор) + axial PCA → `PreAlignParams` (z_min, z_max, dx, dy, rz_pca)
- `crop_z_span()`, `center_crop_yx()` — утилиты для Z-crop и срезов классификатора
- dataclass `PreAlignParams` — все величины в мм и радианах (CW+)

### `pose_regressor.py`

**PoseRegressor3D** — лёгкий 3D CNN (~42k params):

- вход `[B, 1, Z, 72, 72]` после pre-align
- spatial map + masked mean по Z → MLP → (rz, ry, rx) residual
- `from_checkpoint()`, `save_checkpoint()` — загрузка/сохранение весов
- inference углов — через `HeadAligner.predict_pose_angles()` (не отдельный метод модели)

### `head_aligner.py`

**HeadAligner** — полный inference-пайплайн:

- `align_volume()` — Z-crop + shift + только rz_pca @ 4 mm (вход pose)
- `apply_pose_volume()` — prealign + aug (для генерации pose dataset)
- `apply_full_align()` — финальный rigid на 1 mm (rz_pca + rz_pose, ry, rx)
- `AlignResult`, класс `HeadAligner` с `align()` и `from_checkpoints()`

---

## `datasets/`

### `z_slice_head.py`

PyTorch-датасет для Z-классификатора:

- `ZSliceNpyDataset` — `.npy` срезы 56×56, label 0/1
- `list_z_slice_files`, `split_z_slice_files`, `filter_z_slice_dataset_in_place` — для train pipeline
- `collate_slices` — batch collation

### `pose_volume.py`

PyTorch-датасет для pose regressor:

- `PoseSample`, `list_pose_samples`, `split_pose_by_case` — чтение meta JSON + NIfTI
- `PoseVolumeDataset` — загрузка 3D объёма, center crop/pad 72×72, texture aug на train
- `collate_pose_volumes` — variable Z + padding mask

---

## `utils/`

### `__init__.py`

I/O и препроцессинг CT:

- `read_nifti`, `canonicalize_cq500_orientation` — чтение + flip Y при необходимости
- `prepare_isotropic_ct`, `resample_ct_to_isotropic`
- `apply_brain_ct_window` — WL 40/80 → [0, 1]
- `center_crop_yx_to_square`, `center_crop_pad_yx`
- `center_slice_np` — mid-slice для axial/coronal/sagittal MPR

### `angles.py`

Конвенция углов **CW-positive** и маппинг в scipy Euler ZYX:

- `segment_tilt_cw_rad`, `pca_e1_tilt_cw_rad` — из guide-линий и PCA
- `prealign_apply_euler_cw`, `pose_apply_euler_cw`, `full_align_apply_euler_cw` — знаки для apply

### `axial_pca.py`

PCA на пороговой маске аксиального среза:

- `central_slice_axial_pca()` — median center + главная ось e1
- `shift_xy_to_center()` — сдвиг в мм для prealign
- используется в `PreAligner.predict_params()`

### `rigid.py`

3D rigid resampling через SimpleITK:

- `apply_rigid_volume_zyx()` — rotvec + shift в voxels
- `save_volume_nifti`, `load_volume_nifti`, `volume_to_nifti_bytes`
- `compute_full_align_affine_4x4()` — 4×4 rigid input → aligned output
- affine вокруг центра объёма, опционально nearest-neighbor для масок

### `guide_labels.py`

Парсинг ручной JSON-разметки:

- `GuideLabel` dataclass (rot_z, rot_y, rot_x=0, z span)
- `parse_guide_annotation()`, `load_guide_labels()` — JSON → углы + Z-indices @ 4 mm
- `coronal_y_to_z_indices`, `z_1mm_to_cls_indices` — маппинг координат MPR → classifier grid

### `labels.py`

Утилиты имён кейсов:

- `case_id_from_name()` — regex `CQ500CT\d+`
- `load_case_id_list()` — чтение `no_heads.txt`

### `mask_residual.py`

Golden-метрика по маскам **после** HeadAligner:

- `apply_align_to_mask()` — тот же full align к маске глаза/уха
- `compute_mask_residuals()` — rz/ry по L–R глазам и ушам, `rx_om` по sagittal eye_mid→ear_mid
- `abs_deviation_from_horizontal_rad()` — |остаточный наклон|

---

## `app/` (HTTP API)

| Файл | Роль |
|------|------|
| `main.py` | FastAPI: `/health`, `POST /align` → ZIP |
| `align_service.py` | HeadAligner wrapper, сборка `aligned.nii.gz` + `meta.json` |
| `config.py` | env: `ALIGN_DEVICE`, `PRE_ALIGN_CKPT`, `POSE_CKPT`, cls-пороги |

См. [API.md](./API.md).

---

## Связи между модулями

```
scripts/*  →  models/*  →  utils/*
                ↓              ↑
            datasets/*         │
                ↓              │
            paths.py      app/*
```

**Inference (без обучения):** `HeadAligner` + `paths` + `utils` (I/O, angles, rigid).  
**HTTP:** `app/` → `HeadAligner`. Пакет `datasets/` нужен только для train-скриптов.
