# HTTP API

FastAPI-сервис в `app/`. Тот же **HeadAligner**, что в офлайне.

## Запуск

```bash
# веса: weights/pre_aligner_best.pt, weights/pose_regressor_best.pt
docker compose up --build -d
```

GPU (Linux + nvidia-container-toolkit):

```bash
ALIGN_DEVICE=cuda docker compose up -d
```

Локально:

```bash
pip install -r requirements-api.txt
export ALIGN_DEVICE=cuda   # или cpu
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Эндпоинты

### `GET /health`

```json
{
  "status": "ok",
  "device": "cuda",
  "pre_align_ckpt": "/app/weights/pre_aligner_best.pt",
  "pose_ckpt": "/app/weights/pose_regressor_best.pt"
}
```

### `POST /align`

**Request:** `multipart/form-data`

| Поле | Тип | Описание |
|------|-----|----------|
| `file` | file | входной `.nii.gz` / `.nii` |
| `case_id` | string, optional | id кейса для meta |

**Response:** `application/zip`

| Файл в ZIP | Содержимое |
|------------|------------|
| `aligned.nii.gz` | выровненная голова @ 1 mm (если `status=ok`) |
| `meta.json` | полный JSON как в `align/meta/{case}.json` |

**Headers ответа:**

- `X-Align-Status` — `ok` | `no_head` | `pre_align_fail` | …
- `X-Case-Id`

### `meta.json` (ключевые поля)

```json
{
  "case_id": "CQ500CT48",
  "status": "ok",
  "has_head": true,
  "prealign": { "z_min", "z_max", "dx", "dy", "rz_pca_rad", "rz_pca_deg" },
  "pose": { "rz_rad", "ry_rad", "rx_rad", "rz_deg", "ry_deg", "rx_deg" },
  "affine_4x4_input_to_output": [[...], ...],
  "affine_note": "physical XYZ, prepared 1mm input → aligned output"
}
```

`affine_4x4_input_to_output` — матрица 4×4 (row-major list), та же геометрия, что `apply_full_align`.

## Golden eval через API

```bash
python -u scripts/run_head_align_golden.py --service-url http://localhost:8000
```

Пишет в `data/cq500_test/align/` — те же `volumes/`, `meta/`, `results.csv`, что офлайн.

Офлайн fallback:

```bash
python -u scripts/run_head_align_golden.py --offline --device cuda
```

## Модули `app/`

| Файл | Роль |
|------|------|
| `main.py` | FastAPI app, `/health`, `/align` |
| `align_service.py` | обёртка HeadAligner, сборка ZIP |
| `config.py` | env: device, пути к весам, cls-пороги |
