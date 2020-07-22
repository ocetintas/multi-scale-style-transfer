# ------------------------------------------------------------------------------
# First architecture implementation with adversarial and content reconstruction loss.
# Discriminator -> Discriminates STYLE and GENERATOR OUTPUT
# Content reconstruction loss -> Content is fed as a style input and L2 loss is calculated between G output and original content
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utilities import adain
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('Conv2D',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('BatchNorm',nn.BatchNorm2d(out_channel, track_running_stats=False)),
        self.add_module('LeakyReLU',nn.LeakyReLU(0.2, inplace=True))


class ConvBlockNormless(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlockNormless,self).__init__()
        self.add_module('Conv2D',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('LeakyReLU',nn.LeakyReLU(0.2, inplace=True))


# Same with SinGAN
class Discriminator(nn.Module):
    """
    Fully convolutional discriminator (with stride 1) where the feature channel size changes as follows:
        Head: opt.nc_im -> opt.nfc
        Body: opt.nfc / 2^i ->  opt.nfc / 2^(i-1)
                Halving the number of feature channels until opt.min_nfc
        Tail: max(opt.min_nfc, opt.nfc / 2^(opt.num_layer-2)) -> 1
    """

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)

        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('ConvBlock%d' % (i + 1), block)

        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

    @staticmethod
    def initialize_weight(module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            # nn.init.constant_(module.weight, 1)
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)


# MODIFIED GENERATOR
# Generator has 2 input, one coming from style input and the other coming from lower layers
class Generator(nn.Module):
    """
    Fully convolutional generator (with stride 1) where the feature channel size changes as follows:
        Head: opt.nc_im -> opt.nfc
        Body: opt.nfc / 2^i ->  opt.nfc / 2^(i-1)
                Halving the number of feature channels until opt.min_nfc
        Tail: max(opt.min_nfc, opt.nfc / 2^(opt.num_layer-2)) -> 1

    The output of the generator's forward pass is ?
    """

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc

        # STYLE
        self.style_head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)

        self.style_body = nn.Sequential()
        for i in range(opt.num_layer-2):
            block = ConvBlock(N,N,opt.ker_size,opt.padd_size,1)
            self.style_body.add_module('ConvBlock%d'%(i+1),block)

        self.style_tail = nn.Conv2d(N,N,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

        # CONTENT
        # Head (After head there will be adain layer)
        self.content_head = nn.Sequential()
        self.content_head.add_module('ConvBlock_input', ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1))
        self.content_head.add_module('ConvBlock_ConvBlock1', ConvBlock(N, N, opt.ker_size, opt.padd_size, 1))

        # Body
        self.content_body = nn.Sequential()
        self.content_body.add_module('ConvBlock_ConvBlock2', ConvBlock(N, N, opt.ker_size, opt.padd_size, 1))
        self.content_body.add_module('ConvBlock_ConvBlock3', ConvBlock(N, N, opt.ker_size, opt.padd_size, 1))

        # Tail
        self.content_tail = nn.Sequential(
            nn.Conv2d(N,opt.nc_im,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y, style):
        # Forward pass of the style
        style = self.style_head(style)
        style = self.style_body(style)
        style = self.style_tail(style)

        # Forward pass of the content
        x = self.content_head(x)
        x = adain(x, style)     # AdaIN Layer 1
        x = self.content_body(x)
        x = adain(x, style)     # AdaIN Layer 2
        x = self.content_tail(x)

        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

    @staticmethod
    def initialize_weight(module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            # nn.init.constant_(module.weight, 1)
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)

