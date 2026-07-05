# CQ500 head alignment

Двухстадийный ML-пайплайн: **PreAligner** (Z-span + axial PCA) + **PoseRegressor3D** (остаточные rz, ry, rx) → выровненная голова @ 1 mm.

**Отчёт по исследованию:** [REPORT.md](./REPORT.md)  
**Справочник по скриптам:** [docs/SCRIPTS.md](./docs/SCRIPTS.md)  
**Справочник по модулям:** [docs/MODULES.md](./docs/MODULES.md)

> HTTP API, Docker и экспорт affine — **следующий этап** (см. конец отчёта).

---

## Установка

```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Нужен GPU для обучения и golden eval (`--device cuda`). CPU — только для разметки и экспорта labels.

---

## Что за что отвечает

| Часть репозитория | Назначение |
|-------------------|------------|
| `paths.py` | Все пути к данным и константы (spacing 4 mm / 1 mm, веса) |
| `models/` | Нейросети и inference: Z-классификатор, PCA-prealign, pose CNN, **HeadAligner** |
| `datasets/` | PyTorch-датасеты для train Z-cls и pose |
| `utils/` | I/O NIfTI, углы, rigid-трансформы, парсинг guide-разметки, mask-eval |
| `scripts/` | CLI: разметка → train → golden → метрики |
| `weights/` | Чекпоинты `pre_aligner_best.pt`, `pose_regressor_best.pt` (после обучения) |
| `data/` | Volumes и маски локально; в git — только лёгкие артефакты (см. ниже) |

**Точка входа inference:**

```python
import torch
from models.head_aligner import HeadAligner

aligner = HeadAligner.from_checkpoints(device=torch.device("cuda"))
result = aligner.align("path/to/scan.nii.gz", device=torch.device("cuda"))
# result.volume_aligned_1mm  — numpy [Z,Y,X] @ 1 mm, brain window
# result.rz_pca_rad, result.rz_pose_rad, result.ry_pose_rad, result.rx_pose_rad
```

---

## Раскладка данных

Volumes и маски **в git не коммитим** — скачиваются отдельно ([ссылка в TASK.md](./TASK.md)).

**В git коммитим** (настроено в `.gitignore`):

| Путь | Содержимое |
|------|------------|
| `data/cq500_train/no_heads.txt` | case_id без головы (негативы Z-cls) |
| `data/cq500_train/cq500_train_guides/` | сырая JSON-разметка после аннотатора |
| `data/train_logs/` | логи и history обучения |
| `data/cq500_test/mask_residual/` | residual angles eval: CSV, summary.json, meta/ |

```
data/
├── cq500_train/                          # обучение
│   ├── volumes/                          # ← NIfTI train (локально, не в git)
│   │   └── CQ500CT*.nii.gz
│   ├── no_heads.txt                      # в git
│   ├── cq500_train_mpr_1mm/              # PNG для разметки (генерируется, не в git)
│   │   ├── manifest.json
│   │   └── CQ500CT*_axial.png, *_coronal.png, *_sagittal.png
│   ├── cq500_train_guides/               # в git — JSON после аннотатора
│   │   └── CQ500CT*.json
│   ├── cq500_train_guides_analysis/      # экспорт углов
│   │   └── guide_labels.json
│   ├── z_head_slice_cls/                 # .npy срезы для Z-классификатора
│   │   ├── positive/
│   │   └── negative/
│   └── pose_dataset/                     # синтетика для pose (генерируется)
│       ├── volumes/
│       └── meta/
│
├── cq500_test/                           # golden / test
│   ├── volumes/                          # ← 100 тестовых NIfTI (локально)
│   ├── masks/                            # ← TotalSegmentator (локально)
│   │   └── CQ500CT*/
│   │       ├── brain_structures/
│   │       └── head_glands_cavities/
│   ├── align/                            # результат HeadAligner
│   │   ├── volumes/
│   │   ├── meta/
│   │   ├── results.csv
│   │   └── previews/
│   └── mask_residual/                    # в git — метрика по маскам глаз/ушей
│
├── train_logs/                           # в git — логи обучения
└── ...
weights/
    ├── pre_aligner_best.pt
    └── pose_regressor_best.pt
```

**Минимум для старта разметки:** train volumes локально в `data/cq500_train/volumes/`.  
**В репозиторий:** JSON из `cq500_train_guides/`, `no_heads.txt`, при желании `train_logs/`.

---

## Воспроизведение: порядок команд

Все команды — из **корня репозитория** (где `paths.py`).  
Полный список дублируется в [PIPELINE_COMMANDS.txt](./PIPELINE_COMMANDS.txt).

### 0. Подготовка данных

1. Скачать датасет из ТЗ, распаковать.
2. Train volumes → `data/cq500_train/volumes/`
3. Test volumes → `data/cq500_test/volumes/`
4. Test masks → `data/cq500_test/masks/` (для eval по маскам)

### 1. Разметка (train)

```bash
# MPR PNG @ 1 mm для кликов в аннотаторе
python -u scripts/render_guide_annotation_mpr.py

# Интерактивная разметка guide-линий → JSON в cq500_train_guides/
python -u scripts/guide_line_annotator.py

# Экспорт углов + Z-span → guide_labels.json
python -u scripts/export_guide_labels.py
```

### 2. Обучение PreAligner (Z-классификатор срезов)

```bash
python -u scripts/train_z_head_slice_cls.py --device cuda
# → weights/pre_aligner_best.pt
```

### 3. Генерация pose dataset + обучение pose

```bash
python -u scripts/generate_pose_dataset.py --device cuda
python -u scripts/train_pose.py --device cuda --epochs 100 --batch-size 4 --val-frac 0.2 --lr 1e-3
# → weights/pose_regressor_best.pt
```

### 4. Golden eval (test, 100 кейсов)

```bash
docker compose up --build -d
python -u scripts/run_head_align_golden.py --service-url http://localhost:8000
python -u scripts/render_align_previews.py
```

Без Docker: `python -u scripts/run_head_align_golden.py --offline --device cuda`

Результат: `data/cq500_test/align/volumes/*.nii.gz` @ 1 mm, meta, CSV, PNG before/after.

### 5. Оценка качества по маскам (test)

Нужны предсказания из шага 4 и маски TotalSegmentator.

```bash
python -u scripts/eval_mask_residual_angles.py
```

Результат: `data/cq500_test/mask_residual/summary.json` — остаточный наклон линий глаз/ушей **после** align.

---

## HTTP API и Docker

Сервис оборачивает **HeadAligner**: upload NIfTI → ZIP с `aligned.nii.gz` + `meta.json` (углы, affine 4×4).

```bash
docker compose up --build -d
curl http://localhost:8000/health
curl -X POST http://localhost:8000/align \
  -F "file=@data/cq500_test/volumes/CQ500CT48.nii.gz" \
  -F "case_id=CQ500CT48" \
  -o CQ500CT48_align.zip
```

В ZIP:
- `aligned.nii.gz` — голова @ 1 mm (brain window, как в офлайн golden)
- `meta.json` — `prealign`, `pose`, `affine_4x4_input_to_output`, `status`

Переменные окружения: `ALIGN_DEVICE` (`cpu`/`cuda`), `PRE_ALIGN_CKPT`, `POSE_CKPT`.  
Веса монтируются из `./weights` (см. `docker-compose.yml`).

Локально без Docker:

```bash
pip install -r requirements-api.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Подробнее: [docs/API.md](./docs/API.md).

---

## Только inference (уже обученные веса)

```bash
pip install -r requirements.txt
# положить веса в weights/
docker compose up -d
python -u scripts/run_head_align_golden.py --service-url http://localhost:8000
python -u scripts/render_align_previews.py
python -u scripts/eval_mask_residual_angles.py   # опционально, нужны masks
```

---

## Рекомендуемые кейсы из ТЗ

| Кейс | Ожидание |
|------|----------|
| `CQ500CT06`, `CQ500CT243` | уже выровнены |
| `CQ500CT48` | заметный поворот, хороший sanity-check |

Превью: `data/cq500_test/align/previews/{case}_before_after.png`.
