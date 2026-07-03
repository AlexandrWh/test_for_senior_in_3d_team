from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_dicom_series(dicom_dir: str | Path) -> sitk.Image:
    """
    Читает DICOM-серию как 3D SimpleITK Image.
    Обычно для CT SimpleITK применяет RescaleSlope/RescaleIntercept,
    поэтому массив получается в HU.
    """
    dicom_dir = str(dicom_dir)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    # Если серий несколько, берём самую длинную (типичный случай для CQ500: scout + head CT).
    series_id = max(
        series_ids,
        key=lambda sid: len(reader.GetGDCMSeriesFileNames(dicom_dir, sid)),
    )
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)

    reader.SetFileNames(series_file_names)
    image = reader.Execute()

    return image


def read_nifti(path: str | Path) -> sitk.Image:
    """
    Читает NIFTI/NIFTI.GZ как SimpleITK Image.
    Приводит ориентацию к конвенции golden CQ500 (см. canonicalize_cq500_orientation).
    """
    return canonicalize_cq500_orientation(sitk.ReadImage(str(path)))


def canonicalize_cq500_orientation(image: sitk.Image) -> sitk.Image:
    """
    Приводит CT к той же ориентации, что golden NIfTI из data/volumes.

    У golden direction[1,1] (связь index-Y -> physical-Y) отрицательный; у сырых
    DICOM->NIfTI — положительный. Без флипа сагитталь в MPR выглядит зеркально,
    а детектор/pose (обучены на golden) дают плохие результаты.

    Idempotent: тома, уже совпадающие с golden, не меняются.
    """
    if image.GetDirection()[4] > 0:
        return sitk.Flip(image, flipAxes=(False, True, False))
    return image


def reorient_nifti_file(path: str | Path) -> bool:
    """Перезаписывает NIfTI в canonical frame, если нужен Y-flip. Возвращает True если файл изменён."""
    path = Path(path)
    raw = sitk.ReadImage(str(path))
    if raw.GetDirection()[4] <= 0:
        return False
    sitk.WriteImage(canonicalize_cq500_orientation(raw), str(path), useCompression=True)
    return True


def sitk_image_to_numpy(image: sitk.Image) -> np.ndarray:
    """
    SimpleITK Image -> numpy array.
    Важно: форма будет [z, y, x].
    """
    return sitk.GetArrayFromImage(image)


def numpy_mask_to_sitk_with_ct_info(
    mask_np: np.ndarray,
    ct_image: sitk.Image,
    dtype: type = np.uint8,
) -> sitk.Image:
    """
    Создаёт SimpleITK Image из numpy-маски и копирует геометрию из CT.

    mask_np должен быть в форме [z, y, x], то есть как результат
    sitk.GetArrayFromImage(ct_image).
    """
    mask_np = mask_np.astype(dtype)

    mask_image = sitk.GetImageFromArray(mask_np)

    if mask_image.GetSize() != ct_image.GetSize():
        raise ValueError(
            f"Mask size {mask_image.GetSize()} != CT size {ct_image.GetSize()}. "
            f"Check mask_np shape. Expected numpy shape [z, y, x] = "
            f"{ct_image.GetSize()[::-1]}"
        )

    mask_image.CopyInformation(ct_image)

    return mask_image


def save_mask_with_ct_info(
    mask_np: np.ndarray,
    ct_image: sitk.Image,
    output_path: str | Path,
    dtype: type = np.uint8,
) -> None:
    """
    Сохраняет маску в NIFTI/MHA/NRRD и т.д. с геометрией от CT.
    """
    mask_image = numpy_mask_to_sitk_with_ct_info(mask_np, ct_image, dtype=dtype)
    sitk.WriteImage(mask_image, str(output_path))


def apply_brain_ct_window(
    hu: np.ndarray,
    window_level: float = 40.0,
    window_width: float = 80.0,
    output_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Применяет мозговое CT-окно к массиву в HU.

    Дефолтное brain window:
    WL = 40 HU
    WW = 80 HU

    Возвращает float32 массив, нормализованный в output_range.
    Например:
        output_range=(0, 1)   -> float image для модели
        output_range=(0, 255) -> можно привести к uint8 для визуализации
    """
    hu = hu.astype(np.float32)

    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0

    clipped = np.clip(hu, lower, upper)

    out_min, out_max = output_range
    windowed = (clipped - lower) / (upper - lower)
    windowed = windowed * (out_max - out_min) + out_min

    return windowed.astype(np.float32)


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkLinear,
    default_value: float = -1024.0,
) -> sitk.Image:
    """
    Ресемплит SimpleITK Image к заданному spacing.

    Важно:
    - image.GetSpacing() задаётся в (x, y, z)
    - image.GetSize() задаётся в (x, y, z)
    - sitk.GetArrayFromImage(image) возвращает numpy-массив в форме [z, y, x]

    Для CT используйте линейную интерполяцию.
    Для масок — ближайший сосед.
    """
    original_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    original_size = np.array(image.GetSize(), dtype=np.int64)
    target_spacing_np = np.array(target_spacing, dtype=np.float64)

    target_size = np.round(original_size * original_spacing / target_spacing_np)
    target_size = np.maximum(target_size, 1).astype(np.int64)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(v) for v in target_spacing_np))
    resampler.SetSize([int(v) for v in target_size])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(image.GetPixelID())

    return resampler.Execute(image)


def resample_ct_to_isotropic(
    ct_image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> sitk.Image:
    """
    Ресемплит CT к изотропному spacing.
    """
    return resample_to_spacing(
        image=ct_image,
        target_spacing=target_spacing,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
    )


def resample_mask_to_isotropic(
    mask_image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> sitk.Image:
    """
    Ресемплит бинарную / меточную маску к изотропному spacing.
    """
    return resample_to_spacing(
        image=mask_image,
        target_spacing=target_spacing,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
    )


def image_center_physical(image: sitk.Image) -> tuple[float, float, float]:
    """
    Физический (мировой) центр объёма в координатах (x, y, z).

    Удобно использовать как центр вращения для sitk.Euler3DTransform
    (или как center для rotation_matrix_to_transform при матрице 3x3):
    если вращать не вокруг центра, голова «уедет» из кадра.
    """
    size = np.array(image.GetSize(), dtype=np.float64)
    center_index = (size - 1.0) / 2.0
    return image.TransformContinuousIndexToPhysicalPoint(
        [float(c) for c in center_index]
    )


def resample_image_with_transform(
    image: sitk.Image,
    transform: sitk.Transform,
    interpolator: int = sitk.sitkLinear,
    default_value: float = -1024.0,
    expand_to_fit: bool = True,
) -> sitk.Image:
    """
    Применяет произвольный (обычно жёсткий sitk.Euler3DTransform) трансформ
    к изображению и ресемплит результат.

    Важно про семантику трансформа в SimpleITK:
    - трансформ задаётся в ФИЗИЧЕСКОМ пространстве (мир, мм), а не в индексах [z, y, x];
    - resampler работает по pull-модели: для каждого вокселя РЕЗУЛЬТАТА трансформ
      говорит, ОТКУДА его взять во ВХОДНОМ изображении (отображение выход -> вход).
      Практическое следствие: если у вас есть преобразование, которое "выпрямляет
      голову" (применяется к кривому CT и делает его прямым), то подать в resampler
      нужно ОБРАТНОЕ к нему (transform.GetInverse()), иначе голова довернётся в ту же
      сторону и станет кривее. Удобнее задавать трансформ через
      rotation_matrix_to_transform(..., invert=True). Направление всегда проверяйте
      глазами на одном примере: правильный вариант ВЫПРЯМЛЯЕТ голову.

    expand_to_fit=True пересчитывает выходную сетку (origin + size) так, чтобы
    повёрнутый объём целиком помещался в кадр и не обрезался. Размер вокселя
    (spacing) и матрица направления (direction) наследуются от входа — поэтому,
    применяя ОДИН и тот же transform к CT и к маске, вы получите согласованные
    сетки (можно безопасно накладывать маску на выровненный CT).

    Для CT используйте sitkLinear и default_value=-1024 (воздух в HU),
    для меток — sitkNearestNeighbor и default_value=0.
    """
    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    # direction в SimpleITK — это 9 чисел в row-major; столбцы матрицы суть
    # единичные векторы осей изображения в мировых координатах
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    out_direction = image.GetDirection()
    out_origin = image.GetOrigin()
    out_size = list(image.GetSize())

    if expand_to_fit:
        size = np.array(image.GetSize(), dtype=np.float64)
        # 8 углов объёма в непрерывных индексах (0 и N по каждой оси)
        corners_idx = [
            (x, y, z)
            for x in (0.0, size[0])
            for y in (0.0, size[1])
            for z in (0.0, size[2])
        ]
        # каждый угол: индекс -> входная физ. точка -> через inverse трансформа
        # туда, куда он попадёт в выходном пространстве
        inv = transform.GetInverse()
        phys = np.array(
            [
                inv.TransformPoint(
                    image.TransformContinuousIndexToPhysicalPoint(idx)
                )
                for idx in corners_idx
            ],
            dtype=np.float64,
        )
        # проекции углов на оси направления (столбцы direction) -> bbox в системе осей
        proj = phys @ direction  # (8, 3): координата вдоль каждой оси
        min_proj = proj.min(axis=0)
        max_proj = proj.max(axis=0)

        out_size = [
            max(int(np.ceil((max_proj[a] - min_proj[a]) / spacing[a])), 1)
            for a in range(3)
        ]
        # мировой origin выходной сетки = direction @ min_proj
        out_origin = tuple((direction @ min_proj).tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([float(v) for v in spacing])
    resampler.SetSize([int(v) for v in out_size])
    resampler.SetOutputDirection(out_direction)
    resampler.SetOutputOrigin(out_origin)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(image.GetPixelID())

    return resampler.Execute(image)


def rotation_matrix_to_transform(
    matrix: np.ndarray,
    center: tuple[float, float, float] | None = None,
    invert: bool = False,
) -> sitk.Transform:
    """
    Заворачивает матрицу поворота / аффинного преобразования в sitk-трансформ,
    пригодный для resample_image_with_transform.

    matrix:
    - 3x3 — только линейная часть (поворот). Обычно нужен center — физический
      центр объёма, вокруг которого крутим (см. image_center_physical).
    - 4x4 — однородная матрица (поворот + сдвиг). Линейная часть и трансляция
      берутся из неё; center в этом случае задавать НЕ нужно (сдвиг уже в матрице).

    Матрица должна быть в ФИЗИЧЕСКОМ пространстве (мир, мм) и в порядке осей
    (x, y, z) — НЕ в индексах вокселя [z, y, x]. Если матрица посчитана в индексах,
    её сначала надо пересчитать через direction/spacing.

    invert=True возвращает обратное преобразование. Какой вариант нужен — зависит
    от того, как ваш алгоритм определил матрицу:
    - матрица "выпрямляет голову", т.е. применяется К содержимому кривого CT и
      делает его прямым (направление вход -> выход) -> invert=True. Так чаще всего
      и бывает, когда вы сами вычислили нужный поворот.
    - матрица уже в формате "выход -> вход" (например, многие алгоритмы регистрации
      возвращают трансформ fixed -> moving) -> invert=False, она готова для resampler.
    Причина инверсии: sitk.Resample работает по pull-модели (для вокселя результата
    ищет источник во входе). Если не уверены — проверьте оба варианта глазами на
    одном примере: правильный ВЫПРЯМЛЯЕТ голову, неправильный доворачивает сильнее.
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    transform = sitk.AffineTransform(3)

    if matrix.shape == (3, 3):
        transform.SetMatrix(matrix.flatten())  # 9 чисел, row-major
        if center is not None:
            transform.SetCenter(center)
    elif matrix.shape == (4, 4):
        transform.SetMatrix(matrix[:3, :3].flatten())
        transform.SetTranslation(matrix[:3, 3].tolist())
        # center намеренно не трогаем: сдвиг уже учтён в матрице
    else:
        raise ValueError(
            f"Ожидается матрица 3x3 или 4x4, получено {matrix.shape}"
        )

    if invert:
        return transform.GetInverse()

    return transform


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # NIFTI CT
    ct_image = read_nifti("/path/to/ct.nii.gz")

    # или DICOM CT
    # ct_image = read_dicom_series("/path/to/dicom_dir")

    # Приводим CT к изотропному spacing (1x1x1 мм), для CT используется линейная интерполяция
    ct_image = resample_ct_to_isotropic(ct_image, target_spacing=(1.0, 1.0, 1.0))

    ct_hu = sitk_image_to_numpy(ct_image)  # shape: [z, y, x]

    # Приводим к мозговому окну и делаем min-max нормализацию.
    # Это вход для модели / удобное представление для визуализации
    # (в этом примере дальше не используется — оставлено как демонстрация).
    ct_brain_window = apply_brain_ct_window(
        ct_hu,
        window_level=40,
        window_width=80,
        output_range=(0, 1),
    )

    # Допустим, у вас есть предсказанная/готовая маска той же формы [z, y, x]
    mask_np = np.zeros_like(ct_hu, dtype=np.uint8)
    mask_np[ct_hu > 40] = 1  # просто игрушечный пример

    save_mask_with_ct_info(
        mask_np=mask_np,
        ct_image=ct_image,
        output_path="/path/to/mask.nii.gz",
        dtype=np.uint8,
    )

    # Если нужно ресемплить уже готовую маску (sitk.Image) к изотропному spacing
    # (в данном случае маска и так в том же spacing, что и ct_image);
    # для масок используется ближайший сосед, чтобы не размывать метки
    mask_image = numpy_mask_to_sitk_with_ct_info(mask_np, ct_image, dtype=np.uint8)
    mask_image = resample_mask_to_isotropic(mask_image, target_spacing=(1.0, 1.0, 1.0))

    # --- Пример выравнивания головы (3D) ---
    
    # Вариант А: алгоритм вернул углы поворота головы (в радианах) вокруг осей x, y, z.
    # Здесь это просто заглушка — вместо неё подставьте свою оценку.
    angle_x, angle_y, angle_z = 0.0, 0.0, 0.0

    # Жёсткий поворот вокруг физического центра объёма
    rotation = sitk.Euler3DTransform()
    rotation.SetCenter(image_center_physical(ct_image))
    rotation.SetRotation(angle_x, angle_y, angle_z)

    # Вариант Б: алгоритм вернул матрицу (3x3 поворот или 4x4 однородную) в
    # физическом пространстве. Тогда вместо блока выше используйте:
    #   rotation_matrix = np.eye(3)  # ваша матрица
    #   rotation = rotation_matrix_to_transform(
    #       rotation_matrix,
    #       center=image_center_physical(ct_image),  # для 3x3; для 4x4 не нужно
    #       invert=True,  # матрица "выпрямляет голову" (вход -> выход);
    #                     # если она уже в формате выход -> вход (напр. из регистрации) — False.
    #                     # Не уверены — прогоните оба варианта и посмотрите, какой выпрямляет.
    #   )

    # Применяем ОДИН и тот же трансформ к CT и к маске -> согласованные выходные сетки.
    # CT: линейная интерполяция, фон -1024 HU; маска: ближайший сосед, фон 0.
    ct_aligned = resample_image_with_transform(
        ct_image,
        rotation,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
    )
    mask_aligned = resample_image_with_transform(
        mask_image,
        rotation,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
    )