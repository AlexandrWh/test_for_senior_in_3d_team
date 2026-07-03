# Отчёт: нормализация ориентации головы на КТ

## 1. Как понял задачу

Нужен **препроцессинг 3D**, который приводит произвольно уложенную голову на КТ к **единой канонической ориентации** по трём осям вращения (roll / yaw / pitch в клиническом смысле: симметрия в коронали, лицо вперёд, без наклона вперёд-назад). Допуск — **±5°** на ось.

На входе — NIfTI HU; на выходе — выровненный объём + **афинная матрица** преобразования. Решение должно быть воспроизводимо (Docker) и иметь **оценку качества**.

Разметки углов поворота в датасете **нет** — только 100 КТ, маски TotalSegmentator и 3 эталонных примера (06, 243 — ок; 48 — кривой).

## 2. Постановка задачи

Выбрана постановка **оценка 3D rigid transform (rotation + частичный shift)** и однократный resample:

1. Отфильтровать сканы без головы (торс, абдомен).
2. Грубо выровнять **axial roll (rotZ)** геометрическим детектором по маске кости/мозга.
3. Зафиксировать FOV на голове (center crop после axial PCA).
4. Предсказать **остаточный 3D поворот** нейросетью.
5. Применить **составной rigid** к HU и вернуть NIfTI + `4×4` affine.

Это полноценное **3D** решение (не 2D-only): pose-модель предсказывает `rotvec ∈ ℝ³`, применяется через SimpleITK `Euler3DTransform` / affine.

Рабочая система координат — **изотропный 4 mm slab** с identity direction (согласовано с обучением). Полный patient-space reorientation — возможное расширение, но не требовалось для согласованности train/infer.

## 3. Target из разметки

**Прямой GT ориентации нет.** Использована **синтетическая супервизия**:

1. Берём кейсы из `ideal_heads_only` (псевдо-выровненный пул CQ500 train).
2. Применяем известный случайный rigid misalignment (`head_align/augment.py`):
   - rotation: uniform geodesic до 15° (positive) / до 45° (negative);
   - translation: только для негативов.
3. GT correction = **обратный** поворот: `correction_params(R, t)` → `rotvec_corr`, `trans_corr_mm`.

Маски TotalSegmentator в обучении **не используются напрямую** — только для отбора train-кейсов и потенциальных экспериментов. Target полностью определяется известным augment.

Негативы cls: сильный misalign + crop из `no_heads` → метка `has_head=0`.

## 4. Попробованные подходы

| Подход | Итог |
|--------|------|
| Dual-end slab + head length (v0) | Нестабильно на длинных FOV (шея/плечи) |
| Iterative pose-only | Blur при цепочке resample; заменён single-pass |
| Coronal PCA / crown span | Убраны; после axial PCA — center crop |
| Largest CC маска | На шее выбирал позвонок; заменено на **все точки > thr** |
| Один аксиальный срез PCA | На длинном FOV попадал в шею; → **3 среза вверх от центра, step 10** |
| Projection pose (3× 2D) | Эксперимент; не в production |
| `expand_to_fit` при export | Терялось XY-смещение; → `expand_to_fit=false` |

## 5. Финальная архитектура

```
4 mm isotropic → brain window → square Y/X → bottom Z=48
→ ClsNet (head / no-head)
→ Axial PCA: 3 slices (z_center + i·10), thr=20, all mask points → shift XY + rotZ
→ Center crop 48×56×56
→ PoseNet 1× forward → residual rotvec
→ composed rigid → single HU resample → NIfTI
```

Детали: [docs/architecture.md](./docs/architecture.md).

## 6. Модели

### FullScanClsNet

- 3D CNN backbone (4 conv blocks + adaptive pool)
- Head: linear → logit `has_head`
- Вход: pre-detector объём `48×56×56` (resize)

### FullScanPoseNet

- Тот же backbone
- Heads: `rotvec` (3), `translation` (3 mm)
- Вход: post-detector crop `48×56×56`
- На inference: применяется **только rotvec** (инверсия предсказанного misalignment)

Чекпоинты: `weights/head_align_cls_best_v2.pt`, `weights/head_align_pose_best_v2.pt`.

## 7. Loss и метрики

### Обучение

| Модель | Loss | Val метрика |
|--------|------|-------------|
| Cls | `BCEWithLogits` | accuracy |
| Pose | **geodesic angle** (rad), `loss=geo` | MAE geodesic (deg) |

Аугментация интенсивности: scale/shift/noise на HU до window.

### Оценка (golden, 100 кейсов)

Скрипт `run_export_golden.py`, без GT-углов:

- **success rate** (cls + detector + export)
- **geodesic_deg** — угол остаточного pose (прокси качества; target ≈ 0)
- **detector_rotz_deg** — величина axial коррекции
- before/after превью (axial / coronal / sagittal)

## 8. Результаты

### Golden holdout (n=100)

| | |
|--|--|
| Успешно | **95** |
| Отказ cls | **5** (CT172, CT230, CT327, CT365, CT477 — торс без головы) |
| geodesic_deg: median | **2.99°** |
| geodesic_deg: mean | **3.68°** |
| geodesic_deg: p95 | **8.41°** |
| geodesic_deg: max | **18.37°** |

Большинство кейсов в допуске ±5°; хвост — сильные исходные наклоны.

### Качественно

- **CQ500CT48** (эталон «нужен поворот») — выравнивается, превью в `data/api_test/`.
- **CQ500CT06 / CT243** — минимальная коррекция.
- **CQ500CT211** — большой axial tilt (~40° detZ) — исправляется.
- **CQ500CT173** — длинный FOV (макушка–плечи): PCA на нижнем slab ближе к шее (известное ограничение bottom slab).

### Val (синтетический)

Логи обучения: `data/train_logs/cls_*.tsv`, `pose_*.tsv` (генерируются при train). Val pose geo MAE для v2 — порядка **2–4°** на синтетике (точное число — по логу best epoch).

## 9. Ограничения и next steps

1. **Train/serve**: `generate_dataset.py` и inference используют один путь `detector_align_slab_pca_zrot` (PCA rotZ + center crop).
2. **FOV export** — infer slab 48×4 mm, не полный скан.
3. **Patient direction** — affine в index frame; для PACS может понадобиться `R_phys = D · R · Dᵀ`.
4. **Длинный FOV** — PCA от центра slab; bottom slab Z=48 фиксирует нижнее окно.

## 10. Запуск

См. [README.md](./README.md): Docker, API, обучение, golden eval.

## 11. Инструменты

- Python, PyTorch, SimpleITK, FastAPI
- Cursor / Claude — помощь в коде и отладке
- CQ500, TotalSegmentator — внешние данные (см. TASK.md)
