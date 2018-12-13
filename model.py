import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt

def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2) * weights.size(3)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))


def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)

class FourConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourConvResBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels, affine=True)
        self.norm2 = nn.BatchNorm2d(in_channels, affine=True)
        self.norm4 = nn.BatchNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        out2 = self.norm4(out2)
        out2 = self.relu(out2)
        out2 = self.conv4(out2)
        out  = x + out1 + out2
        return out
    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))

class FourDilateConvResBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation2, dilation4):
        super(FourDilateConvResBlockBN, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels, affine=True)
        self.norm2 = nn.BatchNorm2d(in_channels, affine=True)
        self.norm4 = nn.BatchNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,  out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels,  out_channels, (3, 3), (1, 1), (dilation2, dilation2), (dilation2, dilation2), bias=False)
        self.conv4 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (dilation4, dilation4), (dilation4, dilation4), bias=False)
    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        out2 = self.norm4(out2)
        out2 = self.relu(out2)
        out2 = self.conv4(out2)
        out  = x + out1 + out2
        return out
    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))

class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoConvBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels, affine=True)
        self.norm2 = nn.BatchNorm2d(in_channels, affine=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)

    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)


        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)

        out  = out1 + out2
        return out
    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))

class reflection(nn.Module):
    def __init__(self):
        super(reflection, self).__init__()
        self.conv0  = nn.Conv2d( 16,  48, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv1  = nn.Conv2d( 3,   48, (3, 3), (1, 1), (1, 1), bias=False)
        
        self.norm1 = nn.BatchNorm2d(3, affine=True)
        self.norm2 = nn.BatchNorm2d(48, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2  = nn.Conv2d( 48,  3, (3, 3), (1, 1), (1, 1), bias=False)

        self.pixel_shuffle = nn.PixelShuffle(4)
        
        self.LocalGrad0 = self.make_layer(FourConvResBlock, 48, 48, 3)
        self.LocalGrad1 = self.make_layer3(48, 48)
        self.LocalGrad2 = self.make_layer2(48, 48)
        self.LocalGrad3 = self.make_layer(FourConvResBlock, 48, 48, 3)

        self.fuse0 = self.make_layer(TwoConvBlock, 48, 16, 1)
        self.fuse1 = self.make_layer(TwoConvBlock, 48, 16, 1)
        self.fuse2 = self.make_layer(TwoConvBlock, 48, 16, 1)
        self.fuse3 = self.make_layer(TwoConvBlock, 48, 16, 1)

        self.GlobalGrad = self.make_layer(TwoConvBlock, 48, 48, 1)
    def forward(self, x):
        x0 = x[:,0,:,:].unsqueeze(1)
        x1 = x[:,1,:,:].unsqueeze(1)
        x2 = x[:,2,:,:].unsqueeze(1)
        out0 = pixel_reshuffle(x0, 4)
        out1 = pixel_reshuffle(x1, 4)
        out2 = pixel_reshuffle(x2, 4)

        out0_0 = self.conv0(out0)
        out0_0 = self.LocalGrad0(out0_0)
        out0_1 = self.LocalGrad1(out0_0)
        out0_2 = self.LocalGrad2(out0_1)
        out0_3 = self.LocalGrad3(out0_2)

        out1_0 = self.conv0(out1)
        out1_0 = self.LocalGrad0(out1_0)
        out1_1 = self.LocalGrad1(out1_0)
        out1_2 = self.LocalGrad2(out1_1)
        out1_3 = self.LocalGrad3(out1_2)

        out2_0 = self.conv0(out2)
        out2_0 = self.LocalGrad0(out2_0)
        out2_1 = self.LocalGrad1(out2_0)
        out2_2 = self.LocalGrad2(out2_1)
        out2_3 = self.LocalGrad3(out2_2)

        out0_0 = self.fuse0(out0_0)
        out0_1 = self.fuse1(out0_1)
        out0_2 = self.fuse2(out0_2)
        out0_3 = self.fuse3(out0_3)

        out1_0 = self.fuse0(out1_0)
        out1_1 = self.fuse1(out1_1)
        out1_2 = self.fuse2(out1_2)
        out1_3 = self.fuse3(out1_3)

        out2_0 = self.fuse0(out2_0)
        out2_1 = self.fuse1(out2_1)
        out2_2 = self.fuse2(out2_2)
        out2_3 = self.fuse3(out2_3)

        out0_5 = out0_0 + out0_1 + out0_2 + out0_3
        out1_5 = out1_0 + out1_1 + out1_2 + out1_3
        out2_5 = out2_0 + out2_1 + out2_2 + out2_3
        out0_5 = self.pixel_shuffle(out0_5)
        out1_5 = self.pixel_shuffle(out1_5)
        out2_5 = self.pixel_shuffle(out2_5)

        out = torch.cat( (out0_5, out1_5, out2_5), 1) + x
        out = self.conv1(out)
        out = self.GlobalGrad(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        transmitted = x + out
        return transmitted
    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
    def make_layer2(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockBN(in_channels, out_channels, 1, 2))
        layers.append(FourDilateConvResBlockBN(in_channels, out_channels, 2, 4))
        layers.append(FourDilateConvResBlockBN(in_channels, out_channels, 4, 8))
        return nn.Sequential(*layers)

    def make_layer3(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockBN(in_channels, out_channels, 8, 4))
        layers.append(FourDilateConvResBlockBN(in_channels, out_channels, 4, 2))
        layers.append(FourDilateConvResBlockBN(in_channels, out_channels, 2, 1))
        return nn.Sequential(*layers)