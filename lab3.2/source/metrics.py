import csv
import numpy as np
import torch

from image_processing import mask_bbox


def psnr(prediction, reference, mask=None):
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    if mask is not None:
        difference = difference[mask]
    if not difference.size:
        return float('nan')
    mse = float(np.mean(difference ** 2))
    return float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


class QualityMetrics:
    def __init__(self):
        import lpips
        import piq
        self.device = torch.device('cpu')
        self.lpips = lpips.LPIPS(net='alex', verbose=False).eval()
        self.clip_iqa = piq.CLIPIQA(data_range=1.0).eval()

    def tensor(self, image):
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].to(
            self.device, dtype=torch.float32) / 255.0

    @torch.inference_mode()
    def score(self, prediction, reference):
        result = {'psnr_db': psnr(prediction, reference), 'lpips': float('nan'),
                  'clip_iqa': float('nan')}


        if min(prediction.shape[:2]) < 64:
            return result
        pred, ref = self.tensor(prediction), self.tensor(reference)
        result.update(lpips=float(self.lpips(pred, ref, normalize=True).item()),
                      clip_iqa=float(self.clip_iqa(pred).item()))
        return result

    def rows(self, prediction, reference, evaluation_mask):
        yield {'scope': 'full', **self.score(prediction, reference)}
        bbox = mask_bbox(evaluation_mask)
        if bbox is not None:
            yield {'scope': 'target_bbox', **self.score(prediction[bbox], reference[bbox])}
        yield {'scope': 'target_mask', 'psnr_db': psnr(prediction, reference, evaluation_mask),
               'lpips': float('nan'), 'clip_iqa': float('nan')}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

