import torch
import torch.nn as nn
from models.mmseg_ops import resize
from models.swinspec import SwinSpectralTransformer
from models.uper import UPerHead

class SwinSpecDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = SwinSpectralTransformer( in_channels=1,
                                                 embed_dims=96,
                                                 patch_size=(4, 4, 4),
                                                 window_size=(1, 7, 7),
                                                 window_size_spectral=(9, 1, 1),
                                                 mlp_ratio=4,
                                                 depths=(2, 2, 6, 2),
                                                 num_heads=(3, 6, 12, 24),
                                                 down_sample_stride=(1, 2, 2),
                                                 out_indices=(0, 1, 2, 3),
                                                 qkv_bias=True,
                                                 qk_scale=None,
                                                 patch_norm=True,
                                                 drop_rate=0.,
                                                 attn_drop_rate=0.,
                                                 drop_path_rate=0.1,
                                                 act_cfg=dict(type='GELU'),
                                                 norm_cfg=dict(type='LN'),
                                                 init_cfg=None,
                                                 with_cp=False,
                                                 use_spectral_aggregation='Token')
        self.decode_head = UPerHead(in_channels=[96, 192, 384, 768],
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_classes=3,
                                    norm_cfg=dict(type='BN', requires_grad=True),
                                    align_corners=False)




    def forward(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.backbone(img)
        out = self.decode_head(x)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.decode_head.align_corners)
        return out
