# Данные

## Скачивание

Датасет тестового задания (100 golden + маски TotalSegmentator):

**https://nextcloud.celsus.dev/s/mR2JGoTKjCGYbes**

Источник КТ: [CQ500](https://www.kaggle.com/datasets/crawford/qureai-headct).

## Размещение в репозитории

После распаковки:

```
data/
  volumes/          # 100 golden NIfTI (CQ500CT*.nii.gz)
  masks/            # TotalSegmentator masks per case
  cq500_train/      # 80% train split (для generate_dataset)
    volumes/
    ideal_heads_only/
    no_heads/
```

Папка `data/` в `.gitignore` — в git не коммитится.

## Эталонные примеры (из TASK.md)

| Файл | Смысл |
|------|--------|
| `CQ500CT06.nii.gz` | уже выровнен |
| `CQ500CT243.nii.gz` | выровнен, с патологией |
| `CQ500CT48.nii.gz` | требует поворота |

## Чекпоинты

Веса v2 лежат в репозитории:

```
weights/head_align_cls_best_v2.pt   (~1.2 MB)
weights/head_align_pose_best_v2.pt  (~1.5 MB)
```

При переобучении `train_cls.py` / `train_pose.py` перезаписывают best в `weights/`.

## Артефакты eval

```
data/api_golden/
  aligned/       # выровненные NIfTI (генерируется)
  meta/          # JSON meta per case
  previews/      # before/after PNG
  results.csv    # сводка по 100 кейсам
```
