# Скрипты (`scripts/`)

Каждый файл — CLI-точка входа. Запуск из корня репозитория.

---

## Разметка и экспорт labels

### `render_guide_annotation_mpr.py`

Рендер MPR-превью для ручной разметки: из train NIfTI строит изотропные **1 mm** объёмы (brain window) и сохраняет PNG по плоскостям axial / coronal / sagittal.

- **Вход:** `data/cq500_train/volumes/*.nii.gz`
- **Выход:** `data/cq500_train/cq500_train_mpr_1mm/` + `manifest.json` (slice indices, shapes)
- **Опции:** `--limit N`, `--combined` (один PNG 1×3 на кейс)

---

### `guide_line_annotator.py`

Интерактивный аннотатор: по 2 клика на панель → жёлтая guide-линия на axial, coronal, sagittal.

- **Вход:** MPR PNG из предыдущего шага
- **Выход:** `data/cq500_train/cq500_train_guides/{case_id}.json` (координаты p0/p1, slice_index)
- **Горячие клавиши:** `s` save+next, `u` undo, `d` delete, `q` quit

Это **сырая разметка** — её можно коммитить в репозиторий.

---

### `export_guide_labels.py`

QC ручных guides + экспорт численных targets для обучения.

- **Вход:** JSON из `cq500_train_guides/`, manifest MPR
- **Выход:** `cq500_train_guides_analysis/guide_labels.json` (rot_z, rot_y, rot_x=0, z_lo/z_hi @ 4 mm), CSV со статистикой и skipped cases
- Использует только **axial + coronal** линии; sagittal в target не идёт

---

## Обучение

### `train_z_head_slice_cls.py`

Генерация датасета срезов 56×56 и обучение **ZSliceHeadClsNet** (часть PreAligner).

1. Из `guide_labels.json` — positive срезы внутри Z-span головы, negative — случайные срезы без головы
2. 3 цикла × 5 эпох + in-place filter (отсев «плохих» срезов по prob)
3. Сохраняет `weights/pre_aligner_best.pt`

- **Опции:** `--skip-generate`, `--generate-only`, `--device cuda`

---

### `generate_pose_dataset.py`

Синтетический датасет для pose regressor: на каждый размеченный кейс **30** случайных rigid-аугментаций @ 4 mm.

- Применяет PreAlign + aug, пишет объём и meta с residual labels
- **Вход:** train volumes, `guide_labels.json`, `weights/pre_aligner_best.pt`
- **Выход:** `data/cq500_train/pose_dataset/volumes/`, `meta/`

---

### `train_pose.py`

Обучение **PoseRegressor3D**: weighted L1 по (rz, ry, rx), split 80/20 по `case_id`.

- **Вход:** `pose_dataset/`
- **Выход:** `weights/pose_regressor_best.pt`, логи и кривые в `data/train_logs/`
- **Опции:** `--epochs`, `--batch-size`, `--val-frac`, `--lr`, `--device`

---

## Golden eval и визуализация

### `run_head_align_golden.py`

Прогон **HeadAligner** на test volumes: infer @ 4 mm, apply @ 1 mm.

- **API:** `--service-url http://localhost:8000` (после `docker compose up`) или env `ALIGN_SERVICE_URL`
- **Офлайн:** `--offline --device cuda`
- **Вход:** `data/cq500_test/volumes/`
- **Выход:** `align/volumes/`, `align/meta/{case}.json`, `align/results.csv`

---

### `render_align_previews.py`

PNG **до/после**: raw @ 1 mm vs aligned head @ 1 mm, сетка 2×3 MPR + углы в footer.

- **Вход:** raw volumes + `align/volumes/` + `align/meta/`
- **Выход:** `align/previews/{case}_before_after.png`

---

### `eval_mask_residual_angles.py`

Метрика качества **после** align: те же трансформации применяются к маскам глаз и ушей (TotalSegmentator), считается остаточный наклон линий L–R и eye→ear.

- **Вход:** `align/meta/`, test volumes, `masks/{case}/head_glands_cavities/`
- **Выход:** `mask_residual/residual_angles.csv`, `summary.json`, `meta/{case}.json`
- Не сравнивает pred vs GT — только «сколько не дотянули» после apply

---

## Порядок в пайплайне

```
render_guide_annotation_mpr
  → guide_line_annotator
  → export_guide_labels
  → train_z_head_slice_cls
  → generate_pose_dataset
  → train_pose
  → run_head_align_golden
  → render_align_previews
  → eval_mask_residual_angles
```

См. также [PIPELINE_COMMANDS.txt](../PIPELINE_COMMANDS.txt) и [API.md](./API.md).
