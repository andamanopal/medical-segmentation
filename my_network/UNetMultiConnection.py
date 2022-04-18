import torch
import torch.nn as nn
from torch.nn import init
import functools


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


class StartModule(nn.Module):
    def __init__(self, in_channels, num_features, activation=nn.PReLU):
        super(StartModule, self).__init__()
        self.start_module = []
        self.start_module += [ConvBlock(in_channels=in_channels,
                                        out_channels=num_features,
                                        kernel_size=3,
                                        padding_type='zero',
                                        padding=1,
                                        stride=1,
                                        norm_layer=nn.BatchNorm2d,
                                        activation=activation)]
        self.start_module += [ConvBlock(in_channels=num_features,
                                        out_channels=num_features,
                                        kernel_size=3,
                                        padding_type='zero',
                                        padding=1,
                                        stride=1,
                                        norm_layer=nn.BatchNorm2d,
                                        activation=activation)]
        self.start_module += [ConvBlock(in_channels=num_features,
                                        out_channels=num_features,
                                        kernel_size=3,
                                        padding_type='zero',
                                        padding=1,
                                        stride=1,
                                        norm_layer=nn.BatchNorm2d,
                                        activation=activation)]
        self.start_module = nn.Sequential(*self.start_module)

    def forward(self, x):
        return self.start_module(x)


class EncodingModule(nn.Module):
    def __init__(self, in_channels, num_features, activation=nn.PReLU):
        super(EncodingModule, self).__init__()
        self.res_block = ConvBlock(in_channels=in_channels,
                                   out_channels=num_features,
                                   kernel_size=3,
                                   padding_type='zero',
                                   padding=1,
                                   stride=2,
                                   norm_layer=nn.BatchNorm2d,
                                   activation=activation)

        self.encoding_block = []
        self.encoding_block += [nn.Dropout2d(p=0.3)]
        self.encoding_block += [ConvBlock(in_channels=in_channels,
                                          out_channels=num_features,
                                          kernel_size=3,
                                          padding_type='zero',
                                          padding=1,
                                          stride=2,
                                          norm_layer=nn.BatchNorm2d,
                                          activation=activation)]
        self.encoding_block += [ConvBlock(in_channels=num_features,
                                          out_channels=num_features,
                                          kernel_size=3,
                                          padding_type='zero',
                                          padding=1,
                                          stride=1,
                                          norm_layer=nn.BatchNorm2d,
                                          activation=activation)]
        self.encoding_block += [ConvBlock(in_channels=num_features,
                                          out_channels=num_features,
                                          kernel_size=3,
                                          padding_type='zero',
                                          padding=1,
                                          stride=1,
                                          norm_layer=nn.BatchNorm2d,
                                          activation=activation)]
        self.encoding_block = nn.Sequential(*self.encoding_block)

    def forward(self, x):
        return self.encoding_block(x) + self.res_block(x)


class DecodingModule(nn.Module):
    def __init__(self, in_channels, num_features, bottom, activation=nn.PReLU):
        super(DecodingModule, self).__init__()
        self.bottom = bottom
        self.upsampling_block = []
        if not self.bottom:
            self.upsampling_block += [nn.Dropout2d(p=0.3)]
            self.upsampling_block += [nn.Upsample(scale_factor=2, mode='bilinear')]
        self.upsampling_block = nn.Sequential(*self.upsampling_block)

        self.decoding_block = []
        self.decoding_block += [ConvBlock(in_channels=in_channels,
                                          out_channels=num_features,
                                          kernel_size=3,
                                          padding_type='zero',
                                          padding=1,
                                          stride=1,
                                          norm_layer=nn.BatchNorm2d,
                                          activation=activation)]
        self.decoding_block += [ConvBlock(in_channels=num_features,
                                          out_channels=num_features,
                                          kernel_size=3,
                                          padding_type='zero',
                                          padding=1,
                                          stride=1,
                                          norm_layer=nn.BatchNorm2d,
                                          activation=activation)]
        self.decoding_block += [ConvBlock(in_channels=num_features,
                                          out_channels=num_features,
                                          kernel_size=3,
                                          padding_type='zero',
                                          padding=1,
                                          stride=1,
                                          norm_layer=nn.BatchNorm2d,
                                          activation=activation)]
        self.decoding_block = nn.Sequential(*self.decoding_block)

        self.res_block = []
        self.res_block += [nn.Upsample(scale_factor=2, mode='bilinear')]
        self.res_block += [ConvBlock(in_channels=num_features,
                                     out_channels=num_features,
                                     kernel_size=3,
                                     padding_type='zero',
                                     padding=1,
                                     stride=1,
                                     norm_layer=nn.BatchNorm2d,
                                     activation=activation)]
        self.res_block = nn.Sequential(*self.res_block)

    def forward(self, x, res_inputs):

        if self.bottom:
            _x = x
        else:
            _x = self.upsampling_block(x)

        for unscaled_input in res_inputs:
            scale_factor = _x.shape[-1] / unscaled_input.shape[-1]
            _x = torch.cat([_x, nn.functional.interpolate(unscaled_input, scale_factor=scale_factor)], dim=1)

        if self.bottom:
            return self.decoding_block(_x)
        else:
            return self.decoding_block(_x) + self.res_block(x)


class EndModule(nn.Module):
    def __init__(self, in_channels, num_features, out_channels, activation=nn.PReLU):
        super(EndModule, self).__init__()
        self.end_module = []
        self.end_module += [ConvBlock(in_channels=in_channels,
                                      out_channels=num_features,
                                      kernel_size=3,
                                      padding_type='zero',
                                      padding=1,
                                      stride=1,
                                      norm_layer=nn.BatchNorm2d,
                                      activation=activation)]
        self.end_module += [ConvBlock(in_channels=num_features,
                                      out_channels=num_features,
                                      kernel_size=3,
                                      padding_type='zero',
                                      padding=1,
                                      stride=1,
                                      norm_layer=nn.BatchNorm2d,
                                      activation=activation)]
        self.end_module += [nn.Conv2d(in_channels=num_features, out_channels=out_channels, kernel_size=(1, 1))]
        self.end_module += [nn.Softmax(dim=1)]
        self.end_module = nn.Sequential(*self.end_module)

    def forward(self, x):
        return self.end_module(x)


class UNetMultiConnection(nn.Module):
    def __init__(self, in_channels, out_channels, num_features=8, upsample_mode="bilinear", activation=nn.PReLU):
        super(UNetMultiConnection, self).__init__()

        self.start_module = StartModule(in_channels=in_channels, num_features=num_features, activation=activation)
        self.encode1 = EncodingModule(in_channels=num_features, num_features=num_features, activation=activation)
        self.encode2 = EncodingModule(in_channels=num_features, num_features=num_features, activation=activation)
        self.encode3 = EncodingModule(in_channels=num_features, num_features=num_features, activation=activation)
        self.encode4 = EncodingModule(in_channels=num_features, num_features=num_features, activation=activation)
        self.encode5 = EncodingModule(in_channels=num_features, num_features=num_features, activation=activation)
        self.encode6 = EncodingModule(in_channels=num_features, num_features=num_features, activation=activation)

        self.decode1 = DecodingModule(in_channels=2*num_features, num_features=num_features, bottom=False, activation=activation)
        self.decode2 = DecodingModule(in_channels=3*num_features, num_features=num_features, bottom=False, activation=activation)
        self.decode3 = DecodingModule(in_channels=4*num_features, num_features=num_features, bottom=False, activation=activation)
        self.decode4 = DecodingModule(in_channels=5*num_features, num_features=num_features, bottom=False, activation=activation)
        self.decode5 = DecodingModule(in_channels=6*num_features, num_features=num_features, bottom=False, activation=activation)
        self.decode6 = DecodingModule(in_channels=7*num_features, num_features=num_features, bottom=False, activation=activation)

        self.end_module = EndModule(in_channels=num_features, num_features=num_features, out_channels=out_channels, activation=activation)

    def forward(self, x):
        enc0 = self.start_module(x)
        enc1 = self.encode1(enc0)
        enc2 = self.encode2(enc1)
        enc3 = self.encode3(enc2)
        enc4 = self.encode4(enc3)
        enc5 = self.encode5(enc4)
        enc6 = self.encode6(enc5)

        dec = self.decode6(enc6, [enc5, enc4, enc3, enc2, enc1, enc0])
        dec = self.decode5(dec, [enc4, enc3, enc2, enc1, enc0])
        dec = self.decode4(dec, [enc3, enc2, enc1, enc0])
        dec = self.decode3(dec, [enc2, enc1, enc0])
        dec = self.decode2(dec, [enc1, enc0])
        dec = self.decode1(dec, [enc0])

        y = self.end_module(dec)
        return y
