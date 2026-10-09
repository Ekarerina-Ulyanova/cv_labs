from pathlib import Path

import argparse
import cv2
import numpy as np
import time


def chroma_key_opencv(foreground, background, hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    result = foreground.copy()
    cv2.copyTo(background, mask, result)

    return result


def chroma_key_native(foreground, background, hsv, lower, upper):
    height, width = foreground.shape[:2]

    hsv_data = hsv.tobytes()
    bg_data = background.tobytes()
    result = bytearray(foreground.tobytes())

    h_min, s_min, v_min = lower
    h_max, s_max, v_max = upper

    for i in range(0, len(hsv_data), 3):
        h, s, v = hsv_data[i:i + 3]

        if (h_min <= h <= h_max
                and s_min <= s <= s_max
                and v_min <= v <= v_max):
            result[i:i + 3] = bg_data[i:i + 3]

    return np.frombuffer(result, dtype=np.uint8).reshape(height, width, 3)


def load_image(path):
    path = Path(path).expanduser().resolve()
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("foreground", help="Путь к основному изображению")
    parser.add_argument("background", help="Путь к фоновому изображению")
    parser.add_argument(
        "lower", nargs=3, type=int, metavar=("H_MIN", "S_MIN", "V_MIN"),
        help="Нижняя граница HSV"
    )
    parser.add_argument(
        "upper", nargs=3, type=int, metavar=("H_MAX", "S_MAX", "V_MAX"),
        help="Верхняя граница HSV"
    )

    args = parser.parse_args()

    lower = np.array(args.lower, dtype=np.int32)
    upper = np.array(args.upper, dtype=np.int32)

    limits = [179, 255, 255]

    for i in range(3):
        if not (0 <= lower[i] <= upper[i] <= limits[i]):
            parser.error("Некорректный диапазон HSV. H: 0–179, S: 0–255, V: 0–255.")

    foreground = load_image(args.foreground)
    background = load_image(args.background)

    h, w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    scale = max(w / bg_w, h / bg_h)

    new_w, new_h = round(bg_w * scale), round(bg_h * scale)
    background = cv2.resize(background, (new_w, new_h))

    x, y = (new_w - w) // 2, (new_h - h) // 2
    background = background[y:y + h, x:x + w]

    hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

    start = time.perf_counter()
    result_opencv = chroma_key_opencv(
        foreground, background, hsv, lower, upper
    )
    time_opencv = time.perf_counter() - start

    start = time.perf_counter()
    result_native = chroma_key_native(
        foreground, background, hsv, lower, upper
    )
    time_native = time.perf_counter() - start

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(results_dir / "opencv.png"), result_opencv)
    cv2.imwrite(str(results_dir / "native.png"), result_native)

    print("\nВремя выполнения:")
    print(f"  OpenCV: {time_opencv:.6f} с")
    print(f"  Собственная реализация: {time_native:.6f} с")

    print(f"Изображения сохранены в: {results_dir}")


if __name__ == "__main__":
    main()