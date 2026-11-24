from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

DEGRADATIONS = {
    'bicubic': (0.0, 0.0, None),
    'blur_noise_jpeg': (1.2, 5.0, 50),
    'severe': (3.2, 18.0, 15),
}


def list_images(directory):
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
    if not paths:
        raise ValueError(f'No images in {directory}')
    return paths


def load_hr(path, size):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        if min(image.size) < size:
            raise ValueError(f'{path}: image {image.size} smaller than requested crop {size}')
        x, y = (image.width - size) // 2, (image.height - size) // 2
        return np.array(image.crop((x, y, x + size, y + size)))


def degrade(hr, name, seed):
    blur, noise, jpeg = DEGRADATIONS[name]
    image = Image.fromarray(hr)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    image = image.resize((image.width // 4, image.height // 4), Image.Resampling.BICUBIC)
    if noise:
        pixels = np.asarray(image, dtype=np.float32)
        pixels = pixels + np.random.default_rng(seed).normal(0, noise, pixels.shape)
        image = Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8))
    if jpeg is not None:
        with BytesIO() as buffer:
            image.save(buffer, format='JPEG', quality=jpeg, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                image = compressed.convert('RGB')
    return np.array(image)


def validate_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != tuple(shape):
        raise ValueError(f'Mask must have LR shape {tuple(shape)}, got {mask.shape}')
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Mask must contain only 0 and 1 (not 255 or probabilities)')
    return mask.astype(bool, copy=True)


def make_mask(shape, kind='center'):
    h, w = shape
    if kind == 'ones':
        return np.ones(shape, dtype=bool)
    mask = np.zeros(shape, dtype=bool)
    if kind == 'center':
        mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    elif kind == 'left':
        mask[:, :w // 2] = True
    elif kind != 'zeros':
        raise ValueError(f'Unknown mask: {kind}')
    return mask


def upscale_mask(mask, scale=4):
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return (slice(int(ys.min()), int(ys.max()) + 1),
            slice(int(xs.min()), int(xs.max()) + 1))


def composite(restored, known_hr, mask_hr):
    return np.where(mask_hr[..., None], restored, known_hr).astype(np.uint8)


def save_collage(path, rows):
    size, header, label = 256, 28, 24
    canvas = Image.new('RGB', (size * len(rows[0][1]), (size + header + label) * len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (degradation, panels) in enumerate(rows):
        top = row * (size + header + label)
        draw.text((6, top + 5), degradation, fill='black')
        for column, (title, pixels) in enumerate(panels):
            left = column * size
            image = Image.fromarray(pixels).convert('RGB').resize((size, size), Image.Resampling.NEAREST)
            canvas.paste(image, (left, top + header + label))
            draw.text((left + 6, top + label + 5), title, fill='black')
    canvas.save(path, format='PNG')
