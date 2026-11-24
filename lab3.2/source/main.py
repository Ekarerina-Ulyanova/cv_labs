import numpy as np
import torch

from pathlib import Path
from PIL import Image

from image_processing import (
    DEGRADATIONS, composite, degrade, list_images, load_hr,
    make_mask, save_collage, upscale_mask,
)
from metrics import QualityMetrics, write_csv
from model import ResShiftInpainting

ROOT = Path(__file__).resolve().parents[1]
SIZE = 256
SEED = 1234


def main():
    torch.set_num_threads(4)
    images = list_images(ROOT / 'data/DIV2K_valid_HR')
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    names = [path.stem.casefold() for path in images]
    if len(names) != len(set(names)):
        raise ValueError('Image names without extensions must be unique.')
    model = ResShiftInpainting()
    metrics = QualityMetrics()
    mask = make_mask((SIZE // 4, SIZE // 4))
    mask_hr = upscale_mask(mask)
    rows = []
    print(f'ResShift v2, device: {model.device}, images: {len(images)}', flush=True)

    for index, path in enumerate(images):
        hr = load_hr(path, SIZE)
        collage = []
        for degradation in DEGRADATIONS:
            print(f'[{index + 1}/{len(images)}] {path.name}: {degradation}', flush=True)
            lr = degrade(hr, degradation, SEED + index)
            lr_up = np.array(Image.fromarray(lr).resize((SIZE, SIZE), Image.Resampling.BICUBIC))
            results = {'lr_up': lr_up}
            for mode, current_mask in (
                ('ones', np.ones_like(mask)),
                ('zeros', np.zeros_like(mask)),
                ('mixed', mask),
            ):
                known_hr = np.where(upscale_mask(current_mask)[..., None], 0, hr).astype(np.uint8)
                results[mode] = model.restore(lr, known_hr, current_mask, SEED + index)
            results['paste_only'] = composite(results['ones'], hr, mask_hr)
            for mode in ('lr_up', 'ones', 'zeros', 'paste_only', 'mixed'):
                for scores in metrics.rows(results[mode], hr, mask_hr):
                    rows.append({'image': path.name, 'degradation': degradation,
                                 'mode': mode, **scores})
            write_csv(output / 'metrics.csv', rows)
            collage.append((degradation, [
                ('Исходное качественное изображение', hr),
                ('Ухудшенное изображение', lr_up),
                ('Маска (белое - область восстановления)', mask_hr.astype(np.uint8) * 255),
                ('Восстановление только по LR', results['ones']),
                ('Подстановка HR после восстановления', results['paste_only']),
                ('Восстановление с HR-контекстом', results['mixed']),
            ]))
        save_collage(output / f'{path.stem}.png', collage)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
