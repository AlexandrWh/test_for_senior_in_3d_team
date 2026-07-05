# Отчёт: нормализация ориентации головы на КТ

Исследование и реализация препроцессинга для приведения 3D КТ головного мозга к канонической ориентации (задание [TASK.md](./TASK.md)).

**Воспроизведение:** [README.md](./README.md), [PIPELINE_COMMANDS.txt](./PIPELINE_COMMANDS.txt)  
**Код:** [docs/SCRIPTS.md](./docs/SCRIPTS.md), [docs/MODULES.md](./docs/MODULES.md)  
**HTTP API:** [docs/API.md](./docs/API.md)

---

## 1. Как поняли задачу

**Проблема:** часть КТ головного мозга приходит с произвольной укладкой головы. Анатомические оси в 3D-объёме не совпадают между исследованиями → падает точность downstream (детекция, сегментация, измерения).

**Цель:** автоматический препроцессинг, который **в 3D** приводит голову к единой канонической ориентации по трём осям вращения.

**Клинические критерии (из ТЗ, упрощённо):**

| Проекция | Критерий |
|----------|----------|
| Аксиальная | вертикальная ось / серп по центру, нет поворота влево-вправо |
| Корональная | нет наклона к плечу, лево/право симметрично |
| Сагиттальная | орбитомеатальная линия горизонтальна, нет наклона вперёд/назад |

**Допуск из ТЗ:** ±5° по каждой оси.

**Выход для продукта:** выровненный 3D NIfTI головы + affine матрица 4×4 (`affine_4x4_input_to_output` в meta / API).

---

## 2. Какую постановку задачи выбрали

Декомпозиция на **два каскадных этапа** + разный spacing на infer и apply:

```
Raw NIfTI
    │
    ▼  PreAligner @ 4 mm
    │    • бинарный Z-классификатор → z_min, z_max (границы головы)
    │    • axial PCA на Z-crop → dx, dy, rz_pca
    │
    ▼  PoseRegressor3D @ 4 mm (на pre-aligned объёме)
    │    • остаточные углы (rz_pose, ry_pose, rx_pose)
    │
    ▼  HeadAligner apply @ 1 mm
         • один rigid: crop Z + shift + rot(rz_pca+rz_pose, ry, rx)
         • выход: cropped head @ 1 mm isotropic
```

**Почему так:**

- Z-span и грубый rz хорошо решаются **геометрией и слабой разметкой** (coronal line → Z, axial → rz).
- Pose-сеть учится только **остаточной** коррекции на уже выровненном объёме — меньший диапазон углов, проще синтетика.
- Infer @ **4 mm** — скорость и стабильность; финальный продукт @ **1 mm** — для клинических downstream.

**Конвенция углов:** измеренный наклон — **CW-positive** (`utils/angles.py`); apply — scipy Euler ZYX с инверсией знака (отворачиваем объём к канону).

---

## 3. Как получили target из разметки

### 3.1 Ручная разметка

1. `render_guide_annotation_mpr.py` — MPR PNG @ 1 mm (axial, coronal, sagittal).
2. `guide_line_annotator.py` — по 2 точки на плоскость, жёлтая guide-линия.
3. `export_guide_labels.py` — экспорт в `guide_labels.json`.

**Используется в обучении:**

| Плоскость | Что даёт |
|-----------|----------|
| **Axial** | наклон линии → `rot_z` (rz_gt) |
| **Coronal** | наклон линии → `rot_y` (ry_gt); Y-координаты концов → `z_lo`, `z_hi` (span головы) |
| **Sagittal** | **не экспортируется** → `rot_x = 0` |

Углы линий: `segment_tilt_cw_rad()` — отклонение от вертикали в PNG-координатах.  
Z-span: coronal Y → индексы @ 1 mm → пересчёт @ 4 mm для классификатора.

**Объём разметки:** 230 train-кейсов с валидными axial + coronal.

**Негативы Z-классификатора:** `no_heads.txt` + случайные срезы вне span.

### 3.2 Target для PoseRegressor (синтетика)

`generate_pose_dataset.py` на каждый размеченный кейс:

1. PreAligner на raw @ 4 mm → `rz_pca`, `z_min`, `z_max`, `dx`, `dy`.
2. Случайный aug: `rz_aug, ry_aug, rx_aug` ~ U(±0.06π) ≈ ±10.8°.
3. Применение prealign + aug к объёму.
4. **Labels (радианы, CW+):**

```
rz_label = rz_gt + rz_aug - rz_pca
ry_label = ry_gt + ry_aug
rx_label = rx_aug          # rz_gt по сагиттали нет
```

Итого: **230 × 30 = 6900** сэмплов @ 4 mm.

На inference `rz_pca` и `rz_pose` хранятся раздельно; при apply @ 1 mm суммируются в один поворот вокруг Z.

---

## 4. Какие подходы пробовали

| # | Подход | Итог |
|---|--------|------|
| 1 | Только Z-классификатор (span головы) | Работает; без углов недостаточно |
| 2 | Axial PCA для rz без доп. разметки | Стабильный геометрический prior для rz_pca |
| 3 | 2D pose на отдельных срезах | Отказ в пользу 3D volume |
| 4 | Тяжёлый 3D CNN + global average pooling | **Collapse** — val MAE ~5.8°, предсказание ≈ константа |
| 5 | Лёгкий 3D CNN + Z-mix + masked mean по Z | Обучение с 1-й эпохи, val MAE ~1.3° |
| 6 | Финальный NIfTI @ 4 mm | Заменено на **apply @ 1 mm** |
| 7 | Два отдельных infer-скрипта | Объединено в **HeadAligner** |
| 8 | GT углы с raw-масок vs pred | Заменено на **residual после apply** на масках глаз/ушей (нагляднее на больших углах) |

---

## 5. Какую модель обучили

### 5.1 PreAligner (`models/pre_aligner.py`)

**Состав:**

1. **ZSliceHeadClsNet** — бинарный классификатор срезов 56×56 @ 4 mm → Z-span головы в мм.
2. **Axial PCA** (`utils/axial_pca.py`) на центральном срезе Z-crop → `dx`, `dy`, `rz_pca`.

**Обучение:** `train_z_head_slice_cls.py` — генерация positive/negative `.npy`, 3×(5 эпох) + in-place filter.

Чекпоинт: `weights/pre_aligner_best.pt`.

### 5.2 PoseRegressor3D (`models/pose_regressor.py`)

Лёгкий 3D CNN, `base_channels=12`, ~42k параметров:

```
[B,1,Z,72,72] → Conv3d stack → spatial 18×18 → Z-mix → masked Z-mean → MLP(32) → (rz, ry, rx)
```

**Обучение:** `train_pose.py` на `pose_dataset/`.

Чекпоинт: `weights/pose_regressor_best.pt`.

### 5.3 HeadAligner (`models/head_aligner.py`)

Оркестрация inference без обучения: 4 mm infer → pose angles → 1 mm apply.

### 5.4 HTTP API (`app/`)

FastAPI-сервис поверх **HeadAligner**: upload NIfTI → ZIP (`aligned.nii.gz` + `meta.json`).  
Golden eval может идти через HTTP (`run_head_align_golden.py --service-url`) или in-process (`--offline`).  
Docker: `docker compose up --build -d`, веса монтируются из `./weights`.

---

## 6. Какие loss и метрики использовали

### 6.1 PreAligner (Z-классификатор)

| | |
|--|--|
| **Loss** | `BCEWithLogits` |
| **Метрики train/val** | accuracy, precision, recall |
| **Split** | 90/10 по файлам срезов |

### 6.2 PoseRegressor3D

| | |
|--|--|
| **Loss** | weighted L1: веса `(0.4, 0.4, 0.2)` для rz, ry, rx |
| **Метрики** | weighted L1, MAE по каждому углу (градусы) |
| **Split** | 80/20 по **уникальным case_id** (aug одного кейса не в обоих сплитах) |
| **Aug train** | brightness, noise, gamma, blur, sharpness |

### 6.3 Golden test (качество на 100 кейсах)

| Метрика | Описание |
|---------|----------|
| **Статусы align** | ok / no_head / fail |
| **Превью** | raw vs aligned @ 1 mm, 2×3 MPR |
| **Mask residual** (`eval_mask_residual_angles.py`) | после apply: \|наклон\| L–R глаз/ушей (axial→rz, coronal→ry), eye_mid→ear_mid на sagittal (`rx_om`) |

Mask residual — **не сравнение pred с GT**, а остаточная ошибка на анатомии после трансформации.

---

## 7. Результаты на validation

### 7.1 PreAligner — Z-классификатор (лучший цикл, ep 5)

| acc | precision | recall |
|-----|-----------|--------|
| **0.997** | **0.999** | **0.997** |

### 7.2 PoseRegressor3D — epoch 100

| Метрика | Train | Val |
|---------|-------|-----|
| Weighted L1 | 0.0182 | **0.0241** |
| MAE rz | — | **1.41°** |
| MAE ry | — | **1.49°** |
| MAE rx | — | **1.12°** |
| MAE mean | — | **1.34°** |

Val — на синтетических **residual** labels после aug ±10.8°; это не прямой клинический benchmark, но показывает, что сеть не коллапсирует и держит ошибку ≪ допуска ±5° на train-домене.

### 7.3 Golden eval — 100 test cases

| Статус | N |
|--------|---|
| ok | **97** |
| no_head | **3** |
| fail | **0** |

Распределение углов на 97 ok (mean | std | max):

| Угол | mean | std | max \|°\| |
|------|------|-----|----------|
| rz_pca | −0.4° | 8.0° | 41.4° |
| rz_pose | −2.2° | 4.3° | 15.2° |
| ry_pose | −0.1° | 3.1° | 12.6° |
| rx_pose | −0.4° | 1.1° | 5.9° |

Качественно: `data/cq500_test/align/previews/{case}_before_after.png`.  
Sanity-кейсы из ТЗ: `CQ500CT06`, `CQ500CT243` (уже ровные), `CQ500CT48` (нужен поворот).

### 7.4 Mask residual (90/100 кейсов, после align)

| Метрика | mean | median | p90 |
|---------|------|--------|-----|
| rz_eyes | 2.2° | 2.1° | 4.2° |
| rz_ears | 2.2° | 1.9° | 4.3° |
| ry_eyes | 1.6° | 1.1° | 3.7° |
| ry_ears | 1.6° | 1.2° | 3.2° |
| **rx_om** | **10.3°** | **7.4°** | **25.8°** |

**Интерпретация:** rz/ry по глазам и ушам в пределах ~2° — хорошо относительно допуска ±5°. `rx_om` хуже: sagittal не размечали (rx_gt=0), pose слабо опирается на анатомию, proxy «уши» TotalSegmentator грубый.

10 пропусков: 3× `no_head`, 7× маска уха пуста после crop/align.

---

## 8. Деплой и inference

| Способ | Команда / эндпоинт |
|--------|-------------------|
| In-process Python | `HeadAligner.from_checkpoints().align(path)` |
| HTTP API | `POST /align` → ZIP с `aligned.nii.gz` + `meta.json` |
| Docker | `docker compose up -d` |
| Golden batch | `run_head_align_golden.py --service-url` или `--offline` |

Affine 4×4 входит в `meta.json` (`affine_4x4_input_to_output`). Отдельный sidecar-файл — опциональное улучшение.

**Возможные доработки:** GPU profile в docker-compose, отдельный экспорт affine, sagittal-разметка для rx.
