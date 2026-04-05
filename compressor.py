import torch
import torch.nn.functional as F


class FeatureCompressor:
    """
    Compress BEV feature maps for V2X transmission via spatial downsampling
    and optional int8 quantization simulation.
    """

    def __init__(self, compression_ratio: int = 2, quantize: bool = False):
        self.compression_ratio = compression_ratio
        self.quantize = quantize

    def compress(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (1, C, H, W) BEV feature map
        Returns:
            (1, C, H//ratio, W//ratio) compressed tensor
        """
        r = self.compression_ratio
        out = F.avg_pool2d(feat, kernel_size=r, stride=r)
        if self.quantize:
            scale = out.abs().max() / 127.0 + 1e-8
            out = (out / scale).round().clamp(-127, 127) * scale
        return out

    def decompress(self, feat: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Args:
            feat: (1, C, H', W') compressed tensor
            target_size: (H, W) target spatial resolution
        Returns:
            (1, C, H, W) bilinear-upsampled tensor
        """
        return F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
