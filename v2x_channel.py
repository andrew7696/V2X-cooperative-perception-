import random
from typing import Optional
import torch
from compressor import FeatureCompressor
from alignment import align_features
from backbone import BEV_RANGE, BEV_SIZE


class V2XChannel:
    """
    Simulated V2X wireless link from Vehicle A to Vehicle B.

    Pipeline:
      feat_A → compress → [optional dropout] → decompress → align → feat_A_aligned

    Parameters
    ----------
    compression_ratio : int
        Spatial downsampling factor applied before transmission.
    quantize : bool
        If True, simulate int8 quantization during compression.
    dropout_rate : float
        Probability [0, 1] that the entire transmission is lost.
        transmit() returns None on a dropped packet.
    latency_ticks : int
        Number of ticks to delay the transmission (staleness simulation).
        Currently buffered internally; caller receives a delayed feature map.
    """

    def __init__(
        self,
        compression_ratio: int = 2,
        quantize: bool = False,
        dropout_rate: float = 0.0,
        latency_ticks: int = 0,
    ):
        self.compressor = FeatureCompressor(compression_ratio, quantize)
        self.dropout_rate = dropout_rate
        self.latency_ticks = latency_ticks
        self._buffer: list = []   # (feat_compressed, pose_src, pose_dst) tuples

    def transmit(
        self,
        feat_a: torch.Tensor,
        pose_a: dict,
        pose_b: dict,
    ) -> Optional[torch.Tensor]:
        """
        Simulate transmitting Vehicle A's BEV features to Vehicle B.

        Args:
            feat_a:  (1, C, H, W) Vehicle A's BEV feature map
            pose_a:  {'x', 'y', 'heading'} Vehicle A world pose (radians)
            pose_b:  {'x', 'y', 'heading'} Vehicle B world pose (radians)

        Returns:
            (1, C, H, W) aligned feature map in Vehicle B's frame,
            or None if the packet was dropped.
        """
        # Dropout
        if random.random() < self.dropout_rate:
            return None

        # Compress
        compressed = self.compressor.compress(feat_a)

        # Latency buffer
        self._buffer.append((compressed, pose_a, pose_b))
        if len(self._buffer) <= self.latency_ticks:
            return None
        compressed_delayed, pose_a_delayed, pose_b_delayed = self._buffer.pop(0)

        # Decompress back to original spatial size
        target_size = (feat_a.shape[2], feat_a.shape[3])
        decompressed = self.compressor.decompress(compressed_delayed, target_size)

        # Spatial alignment into B's frame
        return align_features(decompressed, pose_a_delayed, pose_b_delayed, BEV_RANGE, BEV_SIZE)
