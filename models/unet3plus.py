import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
    """ Custom function for conv2d:
        Apply 3*3 convolutions with BN and ReLU.
    """
    layers = []
    for i in range(1, n + 1):
        conv = nn.Conv2d(in_channels=in_channels if i == 1 else out_channels, 
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding if padding != 'same' else 'same',
                         bias=not is_bn)  # Disable bias when using BatchNorm
        layers.append(conv)
        
        if is_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if is_relu:
            layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)

def dot_product(seg, cls):
    b, n, h, w = seg.shape
    seg = seg.view(b, n, -1)
    cls = cls.unsqueeze(-1)  # Add an extra dimension for broadcasting
    final = torch.einsum("bik,bi->bik", seg, cls)
    final = final.view(b, n, h, w)
    return final

class UNet3Plus(nn.Module):
    def __init__(self, input_shape, output_channels, deep_supervision=False, cgm=False, training=False):
        super(UNet3Plus, self).__init__()
        self.deep_supervision = deep_supervision
        self.CGM = deep_supervision and cgm
        self.training = training

        self.filters = [64, 128, 256, 512, 1024]
        self.cat_channels = self.filters[0]
        self.cat_blocks = len(self.filters)
        self.upsample_channels = self.cat_blocks * self.cat_channels

        # Encoder
        self.e1 = ConvBlock(input_shape[0], self.filters[0])
        self.e2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[0], self.filters[1])
        )
        self.e3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[1], self.filters[2])
        )
        self.e4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[2], self.filters[3])
        )
        self.e5 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[3], self.filters[4])
        )

        # Classification Guided Module
        self.cgm = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(self.filters[4], 2, kernel_size=1, padding=0),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        ) if self.CGM else None

        # Decoder
        self.d4 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.filters[2], self.cat_channels, n=1),
            ConvBlock(self.filters[3], self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d4_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d3 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.filters[2], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d3_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d2 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d2_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d1 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d1_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.final = nn.Conv2d(self.upsample_channels, output_channels, kernel_size=1) if not self.deep_supervision else None

        # Deep Supervision
        self.deep_sup = nn.ModuleList([
            ConvBlock(self.upsample_channels, output_channels, n=1, is_bn=False, is_relu=False)
            for _ in range(5)
        ]) if self.deep_supervision else None

    def forward(self, x) -> torch.Tensor:
        training = self.training
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        # Classification Guided Module
        if self.CGM:
            cls = self.cgm(e5)
            cls = torch.argmax(cls, dim=1).float()

        # Decoder
        d4 = [
            F.max_pool2d(e1, 8),
            F.max_pool2d(e2, 4),
            F.max_pool2d(e3, 2),
            e4,
            F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=True)
        ]
        d4 = [conv(d) for conv, d in zip(self.d4, d4)]
        d4 = torch.cat(d4, dim=1)
        d4 = self.d4_conv(d4)

        d3 = [
            F.max_pool2d(e1, 4),
            F.max_pool2d(e2, 2),
            e3,
            F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=4, mode='bilinear', align_corners=True)
        ]
        d3 = [conv(d) for conv, d in zip(self.d3, d3)]
        d3 = torch.cat(d3, dim=1)
        d3 = self.d3_conv(d3)

        d2 = [
            F.max_pool2d(e1, 2),
            e2,
            F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=8, mode='bilinear', align_corners=True)
        ]
        d2 = [conv(d) for conv, d in zip(self.d2, d2)]
        d2 = torch.cat(d2, dim=1)
        d2 = self.d2_conv(d2)

        d1 = [
            e1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=16, mode='bilinear', align_corners=True)
        ]
        d1 = [conv(d) for conv, d in zip(self.d1, d1)]
        d1 = torch.cat(d1, dim=1)
        d1 = self.d1_conv(d1)

        # Deep Supervision
        outputs = [self.deep_sup[0](d1)] if self.deep_supervision else [F.softmax(self.final(d1), dim=1)]
        if self.deep_sup and self.CGM and training:
            outputs.extend([
                F.interpolate(self.deep_sup[1](d2), scale_factor=2, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[2](d3), scale_factor=4, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[3](d4), scale_factor=8, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[4](e5), scale_factor=16, mode='bilinear', align_corners=True)
            ])

        # Classification Guided Module
        if self.CGM:
            outputs = [dot_product(out, cls) for out in outputs]
            outputs = [torch.sigmoid(out) for out in outputs] 

        if self.CGM:
            outputs.append(cls)
        
        if (self.deep_supervision or self.CGM) and training:
            return outputs
        else:
            return outputs[0]

if __name__ == "__main__":
    INPUT_SHAPE = [1, 320, 320]
    OUTPUT_CHANNELS = 1

    unet_3P = UNet3Plus(INPUT_SHAPE, OUTPUT_CHANNELS, deep_supervision=False, CGM=False)
    unet_3P_deep_sup = UNet3Plus(INPUT_SHAPE, OUTPUT_CHANNELS, deep_supervision=True, CGM=False)
    unet_3P_deep_sup_cgm = UNet3Plus(INPUT_SHAPE, OUTPUT_CHANNELS, deep_supervision=True, CGM=True)
    print(unet_3P)

    # Example input tensor
    x = torch.randn(1, *INPUT_SHAPE)

    # Forward pass
    output = unet_3P(x)
    print(f"Output shape: {output.shape}")
    
    output = unet_3P_deep_sup(x)
    print(f"Output shape: {output.shape}")
    
    output = unet_3P_deep_sup_cgm(x)
    print(f"Output shape: {output.shape}")