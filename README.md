# Head CT orientation normalization

Автоматическое **3D-выравнивание головы** на КТ: детектор ориентации (heuristics + PCA) + pose-сеть, HTTP API с Docker, экспорт выровненного NIfTI и аффинной матрицы.

Постановка задачи: [TASK.md](./TASK.md).

## Быстрый старт (Docker)

```bash
# 1. Скачать данные — см. docs/data.md (веса уже в weights/)
# 2. docker compose up --build
```

Сервис: **http://localhost:8000**

```bash
curl http://localhost:8000/health
```

## Быстрый старт (локально)

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

pip install -r requirements.txt

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Переменные окружения (опционально):

| Переменная | По умолчанию |
|------------|----------------|
| `HEAD_ALIGN_CLS_CKPT` | `weights/head_align_cls_best_v2.pt` |
| `HEAD_ALIGN_POSE_CKPT` | `weights/head_align_pose_best_v2.pt` |
| `HEAD_ALIGN_DEVICE` | `auto` (cuda если есть) |
| `HEAD_ALIGN_SPACING_MM` | `4.0` |
| `HEAD_ALIGN_CLS_THRESHOLD` | `0.5` |

## API

### `GET /health`

Статус сервиса и описание пайплайна (`pipeline` в JSON).

### `POST /align`

- **Body:** `multipart/form-data`, поле `file` — `.nii` / `.nii.gz`
- **Response:** gzip NIfTI (выровненный infer-slab, HU)
- **Header:** `X-Align-Meta` — JSON с `affine_4x4`, углами, `cls_prob`, `geodesic_deg`, `detector_rotz_deg`, `frame`, `pipeline`

Пример:

```bash
curl -X POST http://127.0.0.1:8000/align \
  -F "file=@data/volumes/CQ500CT48.nii.gz" \
  -o aligned.nii.gz -D -
```

Smoke-тест + превью:

```bash
python scripts/test_align_api.py --input data/volumes/CQ500CT48.nii.gz
# → data/api_test/CQ500CT48_aligned.nii.gz, *_before_after.png
```

### `POST /align-with-meta`

Multipart: `aligned_head.nii.gz` + `meta.json`.

## Оценка на golden (100 кейсов)

```bash
python scripts/run_export_golden.py
python scripts/preview_api_golden.py
```

Итог: `data/api_golden/results.csv`, `aligned/`, `previews/`.

Через HTTP (нужен запущенный сервер):

```bash
python scripts/run_api_golden.py
```

Метрики: [docs/evaluation.md](./docs/evaluation.md). Сводная таблица: [docs/golden_results.csv](./docs/golden_results.csv).

## Воспроизведение обучения

Требуется train split CQ500 в `data/cq500_train/` (см. [docs/data.md](./docs/data.md)).

### 1. Генерация синтетического датасета

Случайный rigid misalignment на «идеальных» головах + негативы (торс без головы):

```bash
python scripts/generate_dataset.py --clean
```

Выход:

- `data/head_align_cls/{train,val}/*.npz`
- `data/head_align_pose/{train,val}/*.npz`

### 2. Обучение cls

```bash
python scripts/train_cls.py --epochs 80 --tag v2
# best → weights/head_align_cls_best_v2.pt
```

Loss: `BCEWithLogits`. Метрика val: accuracy.

### 3. Обучение pose

```bash
python scripts/train_pose.py --epochs 400 --base-channels 32 --loss geo --tag v2
# best → weights/head_align_pose_best_v2.pt
```

Loss: geodesic angle (rad). Метрика val: MAE geodesic (deg).

### 4. Eval / debug

```bash
python scripts/run_pipeline.py --case-id CQ500CT48 --save-previews
python scripts/debug_pipeline.py --step 3 --case-id CQ500CT48
```

## Структура репозитория

```
app/                 # FastAPI
head_align/          # пайплайн, модели, геометрия
weights/             # чекпоинты cls/pose (v2)
scripts/             # dataset, train, eval
docs/                # архитектура, данные, eval
utils.py             # I/O, resample, transforms (см. UTILS.md)
Dockerfile
docker-compose.yml
REPORT.md            # отчёт по заданию
```

## Документация

- [docs/architecture.md](./docs/architecture.md) — дизайн пайплайна
- [docs/data.md](./docs/data.md) — скачивание данных и чекпоинтов
- [docs/evaluation.md](./docs/evaluation.md) — метрики golden / val
- [UTILS.md](./UTILS.md) — хелперы `utils.py`
- [REPORT.md](./REPORT.md) — полный отчёт

## Зависимости

- Python 3.11+
- PyTorch, SimpleITK, FastAPI — см. `requirements.txt`
- Conda: `environment.yml` (рекомендуется `pip install -r requirements.txt` для torch)

## Лицензии и источники

- КТ: CQ500 (Kaggle / qure.ai)
- Маски: [TotalSegmentator](https://github.com/wasserth/totalsegmentator)
- В разработке использовались AI-ассистенты (Cursor / Claude)
