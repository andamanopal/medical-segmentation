import functools

import torch
import torch.nn as nn
from collections import namedtuple


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.relu = nn.PReLU()
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


# Temporary Settings
ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
resnet_config = {'resnet18': ResNetConfig(block=BasicBlock,
                                           n_blocks=[2, 2, 2, 2],
                                           channels=[64, 128, 256, 512]),
                 'resnet34': ResNetConfig(block=BasicBlock,
                                           n_blocks=[3, 4, 6, 3],
                                           channels=[64, 128, 256, 512])}


class ResNet(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.PReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):

        y0 = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        y1 = x

        x = self.maxpool(x)
        x = self.layer1(x)
        y2 = x

        x = self.layer2(x)
        y3 = x

        x = self.layer3(x)
        y4 = x

        x = self.layer4(x)
        y5 = x

        return y5, [y4, y3, y2, y1]


class Encoder(nn.Module):
    def __init__(self, in_channels, resnet_config=resnet_config['resnet18'], pretrained_path=None):
        super().__init__()
        self.resnet_encoder = ResNet(input_channels=in_channels, config=resnet_config)
        if pretrained_path is not None:
            self.weight = torch.load(pretrained_path)
            self.weight = {k: v for k, v in self.weight.items() if k in self.resnet_encoder.state_dict()}
            self.weight['conv1.weight'] = self.resnet_encoder.state_dict()['conv1.weight']

            self.resnet_encoder.load_state_dict(self.weight, strict=False)

    def forward(self, x):
        return self.resnet_encoder(x)


class ConvBlock(nn.Module):
    """ Define a convolution block (conv + norm + actv). """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding_type="zero", padding=1, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.PReLU):
        super(ConvBlock, self).__init__()

        # use_bias setup
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv_block = []
        # Padding option
        p = 0
        if padding_type == "reflect":
            self.conv_block += [nn.ReflectionPad2d(padding)]
        elif padding_type == "replicate":
            self.conv_block += [nn.ReplicationPad2d(padding)]
        elif padding_type == "zero":
            p = padding
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        self.conv_block += [nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=p,
                                      dilation=dilation,
                                      bias=use_bias)]
        self.conv_block += [norm_layer(num_features=out_channels)] if norm_layer is not None else []
        self.conv_block += [activation()] if activation is not None else []
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)


class ConvBlock2(nn.Module):
    """ Define a double convolution block (conv + norm + actv). """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding_type="zero", padding=1, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.PReLU):
        super(ConvBlock2, self).__init__()

        self.conv_block = []
        self.conv_block += [ConvBlock(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, padding_type=padding_type, padding=padding,
                                      dilation=dilation, stride=1,
                                      norm_layer=norm_layer, activation=activation),
                            ConvBlock(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, padding_type=padding_type, padding=padding,
                                      dilation=dilation, stride=stride,
                                      norm_layer=norm_layer, activation=activation)]

        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    """ Define a convolution block with upsampling. """

    def __init__(self, in_channels, out_channels, scale_factor=2,
                 kernel_size=(3, 3), padding=(1, 1), activation=nn.PReLU,
                 mode="nearest"):
        super(UpConv, self).__init__()

        self.up_conv = [nn.Upsample(scale_factor=scale_factor, mode=mode),
                        ConvBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  activation=activation)]
        self.up_conv = nn.Sequential(*self.up_conv)

    def forward(self, x):
        return self.up_conv(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features=32, activation=nn.PReLU, upsample_mode='bilinear',
                 resnet_type='resnet18', weight_path='weights/resnet18-pretrained.pth',):
        super().__init__()
        self.config = resnet_config[resnet_type]
        self.encoder = Encoder(in_channels=in_channels,
                               resnet_config=self.config,
                               pretrained_path=weight_path)

        self.dec_conv4 = ConvBlock2(in_channels=num_features + self.config.channels[-2], out_channels=num_features, activation=activation)
        self.dec_conv3 = ConvBlock2(in_channels=num_features + self.config.channels[-3], out_channels=num_features, activation=activation)
        self.dec_conv2 = ConvBlock2(in_channels=num_features + self.config.channels[-4], out_channels=num_features, activation=activation)
        self.dec_conv1 = ConvBlock2(in_channels=num_features + 64, out_channels=num_features, activation=activation)

        self.up_conv5 = UpConv(in_channels=self.config.channels[-1], out_channels=num_features, mode=upsample_mode, activation=activation)
        self.up_conv4 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode, activation=activation)
        self.up_conv3 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode, activation=activation)
        self.up_conv2 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode, activation=activation)
        self.up_conv1 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode, activation=activation)

        self.conv_1x1 = nn.Conv2d(in_channels=num_features, out_channels=out_channels, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y, skip_connections = self.encoder(x)  # 16, [512, 256, 128, 64, 32]

        dec = self.dec_conv4(torch.cat([self.up_conv5(y), skip_connections[0]], dim=1))
        dec = self.dec_conv3(torch.cat([self.up_conv4(dec), skip_connections[1]], dim=1))
        dec = self.dec_conv2(torch.cat([self.up_conv3(dec), skip_connections[2]], dim=1))
        dec = self.dec_conv1(torch.cat([self.up_conv2(dec), skip_connections[3]], dim=1))
        dec = self.up_conv1(dec)

        y = self.conv_1x1(dec)
        y = self.softmax(y)

        return y
