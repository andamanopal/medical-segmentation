import torch
import torch.nn as nn
from torch.nn import init
import functools
from torchvision import models


class SqExBlock(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            SqExBlock(channels=out_channels)
        )

    def forward(self, x):
        return self.conv_block(x)


class ASPP(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_features,
                      out_channels=num_features,
                      kernel_size=(1, 1),
                      dilation=(1,1), bias=False),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=num_features,
                      out_channels=num_features,
                      kernel_size=(3, 3),
                      dilation=(6,6), padding=6, bias=False),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=num_features,
                      out_channels=num_features,
                      kernel_size=(3, 3),
                      dilation=(12,12), padding=12, bias=False),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=num_features,
                      out_channels=num_features,
                      kernel_size=(3, 3),
                      dilation=(18,18), padding=18, bias=False),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=5 * num_features,
                      out_channels=num_features,
                      kernel_size=(1, 1),
                      bias=False),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        shape = x.shape[-2], x.shape[-1]
        y1 = self.conv_block1(x)
        y1 = nn.Upsample(shape, mode='bilinear')(y1)

        y2 = self.conv_block2(y1)
        y3 = self.conv_block3(y2)
        y4 = self.conv_block4(y3)
        y5 = self.conv_block5(y4)

        y = torch.cat([y1, y2, y3, y4, y5], dim=1)
        return self.conv_block6(y)


def get_vgg_layers(in_channels, config, batch_norm):
    layers = []
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=(3,3), padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


class Encoder1(nn.Module):
    def __init__(self, in_channels, freeze_vgg=True):
        super().__init__()
        vgg19_layers = [[64, 64],
                        ['M', 128, 128],
                        ['M', 256, 256, 256, 256],
                        ['M', 512, 512, 512, 512],
                        ['M', 512, 512, 512, 512]]
        vgg19_bn_layers = dict()
        prev_out_ch = 3

        for index, layer_config in enumerate(vgg19_layers):
            vgg19_bn_layers[f"conv{index}"] = get_vgg_layers(prev_out_ch, layer_config, batch_norm=True)
            prev_out_ch = layer_config[-1]

        vgg_feat_ext = self.VGG(vgg19_bn_layers)

        src_layers = list(
            dict.fromkeys(['.'.join(layer[0].split('.')[:2]) for layer in list(models.vgg19_bn().named_parameters()) \
                           if 'classifier' not in layer[0]]))
        dest_layers = list(
            dict.fromkeys(['.'.join(layer[0].split('.')[:2]) for layer in list(vgg_feat_ext.named_parameters())]))
        mapping = dict(zip(src_layers, dest_layers))

        weight = torch.load('vgg19_bn-c79401a0.pth')

        for key in list(weight):
            try:
                src_layer = '.'.join(key.split('.')[:2])
                dest_layer = mapping[src_layer]
                param_type = key.split('.')[-1]
                weight[f"{dest_layer}.{param_type}"] = weight[key]
            except:
                pass
            del weight[key]

        vgg_feat_ext.load_state_dict(weight)
        print('Loaded Pretrained Weight')

        self.backbone = vgg_feat_ext

        if freeze_vgg:
            print('Freezing VGG19 Layers ....')
            for param in self.backbone.parameters():
                param.requires_grad = False

        if in_channels != 3:
            self.backbone.enc0[0] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3),
                                              padding=(1, 1))

    class VGG(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.enc0 = layers['conv0']
            self.enc1 = layers['conv1']
            self.enc2 = layers['conv2']
            self.enc3 = layers['conv3']
            self.enc4 = layers['conv4']

        def forward(self, x):
            y1 = self.enc0(x)
            y2 = self.enc1(y1)
            y3 = self.enc2(y2)
            y4 = self.enc3(y3)
            y5 = self.enc4(y4)
            return y5, [y1, y2, y3, y4]

    def forward(self, x):
        # Return a tuple of (output, [skip connections])
        return self.backbone(x)


class Decoder1(nn.Module):
    def __init__(self, upsample_mode='bilinear'):
        super().__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.dec0 = ConvBlock(in_channels=1024, out_channels=256)
        self.dec1 = ConvBlock(in_channels=512, out_channels=128)
        self.dec2 = ConvBlock(in_channels=256, out_channels=64)
        self.dec3 = ConvBlock(in_channels=128, out_channels=32)

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate([self.dec0, self.dec1, self.dec2, self.dec3]):
            x = self.upsampler(x)
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = decoder(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1x1(x)
        return self.activation(x)


class Encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc0 = ConvBlock(in_channels=1, out_channels=32)
        self.enc1 = ConvBlock(in_channels=32, out_channels=64)
        self.enc2 = ConvBlock(in_channels=64, out_channels=128)
        self.enc3 = ConvBlock(in_channels=128, out_channels=256)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        y1 = self.enc0(x)
        y2 = self.enc1(self.maxpool(y1))
        y3 = self.enc2(self.maxpool(y2))
        y4 = self.enc3(self.maxpool(y3))
        y5 = self.maxpool(y4)

        return y5, [y1, y2, y3, y4]


class Decoder2(nn.Module):
    def __init__(self, upsample_mode='bilinear'):
        super().__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.dec0 = ConvBlock(in_channels=1024, out_channels=256)
        self.dec1 = ConvBlock(in_channels=128 * 5, out_channels=128)
        self.dec2 = ConvBlock(in_channels=64 * 5, out_channels=64)
        self.dec3 = ConvBlock(in_channels=32 * 5, out_channels=32)

    def forward(self, x, from_enc1, from_enc2):
        from_enc1 = from_enc1[::-1]
        from_enc2 = from_enc2[::-1]
        for i, decoder in enumerate([self.dec0, self.dec1, self.dec2, self.dec3]):
            x = self.upsampler(x)
            x = torch.cat([x, from_enc1[i], from_enc2[i]], dim=1)
            x = decoder(x)
        return x


class DoubleUNet(nn.Module):
    def __init__(self, in_channels, freeze_vgg=True):
        super().__init__()
        self.enc1 = Encoder1(in_channels=in_channels, freeze_vgg=freeze_vgg)
        self.enc2 = Encoder2()
        self.dec1 = Decoder1()
        self.dec2 = Decoder2()
        self.aspp1 = ASPP(num_features=512)
        self.aspp2 = ASPP(num_features=256)
        self.output1 = OutputBlock()
        self.output2 = OutputBlock()

    def forward(self, x):
        original_input = x

        x, skips_1 = self.enc1(x)
        x = self.aspp1(x)
        x = self.dec1(x, skips_1)
        x = self.output1(x)
        x_output1 = x * original_input
        x, skips_2 = self.enc2(x_output1)
        x = self.aspp2(x)
        x = self.dec2(x, skips_1, skips_2)
        pred = self.output2(x)
        bg = 1 - pred

        return torch.cat([bg, pred], dim=1)
