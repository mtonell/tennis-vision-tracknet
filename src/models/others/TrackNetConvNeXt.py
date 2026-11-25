import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

class TrackNetConvNeXt(nn.Module):
    def __init__(self, input_height, input_width, in_channels=9, out_channels=3):
        super().__init__()
        
        # Load ConvNeXt-Tiny backbone
        self.backbone = convnext_tiny(pretrained=False).features
        
        # Modify the stem to accept `in_channels` (9) instead of 3
        self.backbone[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
        
        # Decoder Layers
        # Stage 4 output: 768 channels (16x9)
        # Stage 3 output: 384 channels (32x18)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(768 + 384, 384, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384)
        )
        
        # Stage 2 output: 192 channels (64x36)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(384 + 192, 192, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(192)
        )
        
        # Stage 1 output: 96 channels (128x72)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(192 + 96, 96, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96)
        )
        
        # Final 4x upsampling to 512x288
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Stem -> Stage 1 (128x72)
        e1 = self.backbone[:2](x)
        # Downsample -> Stage 2 (64x36)
        e2 = self.backbone[2:4](e1)
        # Downsample -> Stage 3 (32x18)
        e3 = self.backbone[4:6](e2)
        # Downsample -> Stage 4 (16x9)
        e4 = self.backbone[6:8](e3)
        
        # Decoder
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)
        
        out = self.up4(d3)
        out = self.final_conv(out)
        return out
