import torch
import torch.nn as nn
from models.others.TrackNetV4 import MotionPromptLayer, FusionLayerTypeA, FusionLayerTypeB

def _standard_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )

def _depthwise_conv_block(in_channels, out_channels):
    # First layer must be standard because 9 channels isn't always easily separable into standard groups
    if in_channels == 9:
        return _standard_conv_block(in_channels, out_channels)
        
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels), # Depthwise
        nn.Conv2d(in_channels, out_channels, 1, padding=0),                    # Pointwise
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )

class TrackNetAblation(nn.Module):
    def __init__(self, input_height, input_width, half_channels=False, depthwise=False, fusion_layer_type="TypeA"):
        super(TrackNetAblation, self).__init__()
        self.input_height = input_height
        self.input_width = input_width

        # We keep the advanced TrackNetV4 Motion Prompts
        self.fusion_layer = FusionLayerTypeA() if fusion_layer_type == "TypeA" else FusionLayerTypeB()
        self.motion_prompt = MotionPromptLayer()

        # Decide which convolution block to use based on the 'depthwise' flag
        conv_func = _depthwise_conv_block if depthwise else _standard_conv_block

        # Decide channel width based on 'half_channels' flag
        c64  = 32  if half_channels else 64
        c128 = 64  if half_channels else 128
        c256 = 128 if half_channels else 256
        c512 = 256 if half_channels else 512

        # Encoder
        self.conv1 = conv_func(9, c64)
        self.conv2 = conv_func(c64, c64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = conv_func(c64, c128)
        self.conv4 = conv_func(c128, c128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv5 = conv_func(c128, c256)
        self.conv6 = conv_func(c256, c256)
        self.conv7 = conv_func(c256, c256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv8 = conv_func(c256, c512)
        self.conv9 = conv_func(c512, c512)
        self.conv10 = conv_func(c512, c512)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv11 = conv_func(c512 + c256, c256)
        self.conv12 = conv_func(c256, c256)
        self.conv13 = conv_func(c256, c256)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv14 = conv_func(c256 + c128, c128)
        self.conv15 = conv_func(c128, c128)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv16 = conv_func(c128 + c64, c64)
        self.conv17 = conv_func(c64, c64)
        
        self.conv18 = nn.Sequential(
            nn.Conv2d(c64, 3, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        motion_input = x.view(-1, 3, 3, self.input_height, self.input_width)
        residual_maps, motion_loss = self.motion_prompt(motion_input)

        x1 = self.conv2(self.conv1(x))
        p1 = self.pool1(x1)
        x2 = self.conv4(self.conv3(p1))
        p2 = self.pool2(x2)
        x3 = self.conv7(self.conv6(self.conv5(p2)))
        p3 = self.pool3(x3)
        x4 = self.conv10(self.conv9(self.conv8(p3)))

        u1 = self.up1(x4)
        c1 = torch.cat([u1, x3], dim=1)
        x5 = self.conv13(self.conv12(self.conv11(c1)))

        u2 = self.up2(x5)
        c2 = torch.cat([u2, x2], dim=1)
        x6 = self.conv15(self.conv14(c2))

        u3 = self.up3(x6)
        c3 = torch.cat([u3, x1], dim=1)
        x7 = self.conv17(self.conv16(c3))
        
        x = self.conv18[0](x7)
        x = self.fusion_layer([x, residual_maps])
        out = self.conv18[1](x)
        
        return out, motion_loss
