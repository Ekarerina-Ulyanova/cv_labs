import importlib
import numpy as np
import sys
import torch
import torch.nn.functional as F

from pathlib import Path
from omegaconf import OmegaConf

from image_processing import composite, upscale_mask, validate_mask


def instantiate(config):
    module, name = config.target.rsplit('.', 1)
    return getattr(importlib.import_module(module), name)(**dict(config.get('params', {})))


def load_weights(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


@torch.inference_mode()
def sample_masked(diffusion, model, z_lr, lr, mask_z, z_known=None, progress=None):


    noise = torch.randn_like(z_lr)
    z = diffusion.prior_sample(z_lr, noise)
    for i in reversed(range(diffusion.num_timesteps)):
        t = torch.full((lr.shape[0],), i, dtype=torch.long, device=lr.device)
        if z_known is not None:
            known_t = diffusion.q_sample(z_known, z_lr, t, noise=noise)
            z = torch.where(mask_z, z, known_t)
        z = diffusion.p_sample(model, z, z_lr, t, clip_denoised=False,
                               model_kwargs={'lq': lr})['sample']
        if progress:
            progress(diffusion.num_timesteps - i, diffusion.num_timesteps)
    if z_known is not None:
        z = torch.where(mask_z, z, z_known)
    return z


class ResShiftInpainting:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.repo = root / 'ResShift'
        config_path = self.repo / 'configs/realsr_swinunet_realesrgan256.yaml'
        paths = [config_path, root / 'weights/resshift_realsrx4_s15_v2.pth',
                 root / 'weights/autoencoder_vq_f4.pth']
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.device = torch.device('cpu')
        sys.path.insert(0, str(self.repo))
        config = OmegaConf.load(config_path)

        for module_name in ('models.unet', 'ldm.modules.diffusionmodules.model'):
            module = importlib.import_module(module_name)
            module.XFORMERS_IS_AVAILBLE = False
        config.model.params.use_fp16 = False
        config.autoencoder.params.ddconfig.attn_type = 'vanilla'
        self.model = instantiate(config.model)
        load_weights(self.model, paths[1])
        self.model.eval().requires_grad_(False).to(self.device)
        self.autoencoder = instantiate(config.autoencoder)
        load_weights(self.autoencoder, paths[2])

        self.autoencoder.eval().requires_grad_(False).to(self.device)
        self.diffusion = instantiate(config.diffusion)

    def tensor(self, pixels):
        return torch.from_numpy(np.array(pixels, copy=True)).permute(2, 0, 1)[None].to(
            self.device, dtype=torch.float32) / 127.5 - 1

    @torch.inference_mode()
    def restore(self, lr, known_hr, mask_lr, seed=1234, progress=None):

        mask_lr = validate_mask(mask_lr, lr.shape[:2])
        expected = (lr.shape[0] * 4, lr.shape[1] * 4, 3)
        if known_hr.shape != expected or lr.ndim != 3 or lr.shape[2] != 3:
            raise ValueError(f'Expected RGB HR shape {expected} and RGB LR, got {known_hr.shape}')
        if lr.shape[0] % 64 or lr.shape[1] % 64:
            raise ValueError('LR dimensions must be divisible by 64; use HR crops 256 or 512.')
        if not mask_lr.any():
            return known_hr.copy()
        mask_hr = upscale_mask(mask_lr)
        lr_tensor = self.tensor(lr)
        z_lr = self.diffusion.encode_first_stage(lr_tensor, self.autoencoder, up_sample=True)
        mask_tensor = torch.from_numpy(mask_hr)[None, None].to(self.device, dtype=torch.float32)
        mask_z = F.adaptive_max_pool2d(mask_tensor, z_lr.shape[-2:]).bool()
        z_known = None
        if not mask_lr.all():

            lr_up = F.interpolate(lr_tensor, size=expected[:2], mode='bicubic', align_corners=False)
            canvas = torch.where(mask_tensor.bool(), lr_up, self.tensor(known_hr))
            z_known = self.diffusion.encode_first_stage(canvas, self.autoencoder)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            z = sample_masked(self.diffusion, self.model, z_lr, lr_tensor,
                              mask_z, z_known, progress)
            pixels = self.diffusion.decode_first_stage(z.float(), self.autoencoder)
        if not torch.isfinite(pixels).all():
            raise FloatingPointError('Non-finite model output.')
        pixels = ((pixels[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5)
        result = pixels.round().to(torch.uint8).cpu().numpy()
        return composite(result, known_hr, mask_hr)
