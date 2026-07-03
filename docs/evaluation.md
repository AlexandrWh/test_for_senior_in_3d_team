# Оценка качества

## Golden holdout (100 кейсов)

Скрипт:

```bash
python scripts/run_export_golden.py
python scripts/preview_api_golden.py
```

Результаты: `data/api_golden/results.csv`, превью в `data/api_golden/previews/`.

### Метрики (актуальный пайплайн, hybrid_pca_pose)

| Метрика | Значение |
|---------|----------|
| Успешно выровнено | **95 / 100** |
| Отказ cls (нет головы) | 5: CT172, CT230, CT327, CT365, CT477 |
| Pose residual `geodesic_deg` (ok): mean | **3.68°** |
| Pose residual: median | **2.99°** |
| Pose residual: p95 | **8.41°** |
| Pose residual: max | **18.37°** |
| \|detector rotZ\|: mean | **5.92°** |
| \|detector rotZ\|: p95 | **16.94°** |

Клинический допуск из ТЗ: **±5°** по каждой оси. Большинство кейсов укладываются; хвост p95–max — сложные наклоны / FOV.

## Качественная оценка

- `scripts/debug_pipeline.py` — пошаговые PNG (slab, PCA, pose in/out).
- `scripts/test_align_api.py` — before/after для одного кейса через HTTP.
- ITK-SNAP: window Level=40, Width=90 (см. TASK.md).

## Val при обучении

- **Cls**: accuracy на `data/head_align_cls/val` (лог в `data/train_logs/cls_*.tsv`).
- **Pose**: MAE geodesic angle (deg) на `data/head_align_pose/val` (лог в `data/train_logs/pose_*.tsv`).

Синтетический val не заменяет golden holdout: golden — реальные CQ500 без GT-углов.
