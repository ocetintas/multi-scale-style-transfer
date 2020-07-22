import torch
import torch.nn as nn
from utilities import adain


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('Conv2D',nn.Conv2d(in_channel ,out_channel,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        self.add_module('BatchNorm',nn.BatchNorm2d(out_channel, track_running_stats=False)),
        self.add_module('LeakyReLU',nn.LeakyReLU(0.2, inplace=True))


class SingleBlock(nn.Module):
    def __init__(self, opt):
        super(SingleBlock, self).__init__()
        N = opt.N_baseline
        self.is_cuda = torch.cuda.is_available()

        self.style1_1 = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.style1_2 = ConvBlock(N, N, opt.ker_size, opt.padd_size, 1)
        self.smpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.style2_1 = ConvBlock(N, 2*N, opt.ker_size, opt.padd_size, 1)
        self.style2_2 = ConvBlock(2*N, 2*N, opt.ker_size, opt.padd_size, 1)
        self.smpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.style3_1 = ConvBlock(2*N, 4*N, opt.ker_size, opt.padd_size, 1)
        self.style3_2 = ConvBlock(4*N, 4*N, opt.ker_size, opt.padd_size, 1)

        self.content1_1 = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.content1_2 = ConvBlock(N, N, opt.ker_size, opt.padd_size, 1)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.content2_1 = ConvBlock(N, 2*N, opt.ker_size, opt.padd_size, 1)
        self.content2_2 = ConvBlock(2*N, 2*N, opt.ker_size, opt.padd_size, 1)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.content3_1 = ConvBlock(2*N, 4*N, opt.ker_size, opt.padd_size, 1)
        self.content3_2 = ConvBlock(4*N, 4*N, opt.ker_size, opt.padd_size, 1)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.content4_1 = ConvBlock(4*N, 2*N, opt.ker_size, opt.padd_size, 1)
        self.content4_2 = ConvBlock(2*N, 2*N, opt.ker_size, opt.padd_size, 1)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.content5_1 = ConvBlock(2*N, N, opt.ker_size, opt.padd_size, 1)
        self.content5_2 = nn.Sequential(
            nn.Conv2d(N, opt.nc_im, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x, y, style):
        # Style 1
        style = self.style1_1(style)
        style = self.style1_2(style)
        style_1 = self.smpool1(style)

        # Style 2
        style = self.style2_1(style_1)
        style = self.style2_2(style)
        style_2 = self.smpool2(style)

        # Style 3
        style = self.style3_1(style_2)
        style_3 = self.style3_2(style)

        # Content 1
        x = self.content1_1(x)
        x = self.content1_2(x)
        x = self.mpool1(x)
        x = adain(x, style_1) # adain_1

        # Content 2
        x = self.content2_1(x)
        x = self.content2_2(x)
        x = self.mpool2(x)
        x = adain(x, style_2) # adain_2

        # Content 3
        x = self.content3_1(x)
        x = self.content3_2(x)
        x = adain(x, style_3) # adain_3

        # Content 4
        x = self.up4(x)
        x = self.content4_1(x)
        x = self.content4_2(x)

        # Content 5
        x = self.up5(x)
        x = self.content5_1(x)
        x = self.content5_2(x)

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

