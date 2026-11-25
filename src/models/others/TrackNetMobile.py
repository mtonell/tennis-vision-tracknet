import torch
import torch.nn as nn
import torchvision.models as models
from models.others.TrackNetV4 import MotionPromptLayer, FusionLayerTypeA

class TrackNetMobile(nn.Module):
    def __init__(self, input_height, input_width):
        super(TrackNetMobile, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        
        # Keep TrackNetV4's best features
        self.motion_prompt = MotionPromptLayer()
        self.fusion_layer = FusionLayerTypeA()
        
        # Load pre-trained MobileNetV3-Small
        # It's an ultra-light backbone heavily optimized for speed
        mbv3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = mbv3.features
        
        # Modify the first layer to accept 9 channels (3 frames * 3 colors) instead of 3
        original_conv = self.features[0][0]
        new_conv = nn.Conv2d(9, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
                             stride=original_conv.stride, padding=original_conv.padding, bias=False)
        
        with torch.no_grad():
            # Copy pretrained weights across all 3 frames so it starts with good edge detection
            new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
            new_conv.weight[:, 3:6, :, :] = original_conv.weight.clone()
            new_conv.weight[:, 6:9, :, :] = original_conv.weight.clone()
            # Average the weights to maintain the original activation magnitudes
            new_conv.weight /= 3.0
            
        self.features[0][0] = new_conv
        
        # Group the MobileNetV3 features into Encoder stages
        self.enc1 = self.features[:2]   # Drops to H/4
        self.enc2 = self.features[2:4]  # Drops to H/8
        self.enc3 = self.features[4:9]  # Drops to H/16
        self.enc4 = self.features[9:13] # Drops to H/32
        
        # Dynamically determine the channel sizes from the backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 9, input_height, input_width)
            c1 = self.enc1(dummy).shape[1]
            c2 = self.enc2(self.enc1(dummy)).shape[1]
            c3 = self.enc3(self.enc2(self.enc1(dummy))).shape[1]
            c4 = self.enc4(self.enc3(self.enc2(self.enc1(dummy)))).shape[1]

        # Decoder (U-Net style upsampling but extremely lightweight)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c4 + c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c3)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c2)
        )
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c1)
        )
        
        # Final upsample back to original resolution (H/4 -> H)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        self.final_conv1 = nn.Sequential(
            nn.Conv2d(c1, 3, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Motion Prompt
        motion_input = x.view(-1, 3, 3, self.input_height, self.input_width)
        residual_maps, motion_loss = self.motion_prompt(motion_input)
        
        # 2. Encoder
        x1 = self.enc1(x)  # H/4
        x2 = self.enc2(x1) # H/8
        x3 = self.enc3(x2) # H/16
        x4 = self.enc4(x3) # H/32
        
        # 3. Decoder
        u1 = self.up1(x4)
        c1 = torch.cat([u1, x3], dim=1)
        d1 = self.dec1(c1)
        
        u2 = self.up2(d1)
        c2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(c2)
        
        u3 = self.up3(d2)
        c3 = torch.cat([u3, x1], dim=1)
        d3 = self.dec3(c3)
        
        u4 = self.up4(d3) # Back to HxW
        
        out1 = self.final_conv1(u4)
        
        # 4. Fusion Layer (mixes raw network output with mathematical motion gradients)
        out2 = self.fusion_layer([out1, residual_maps])
        
        return out2, motion_loss
