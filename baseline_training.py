import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
import numpy as np

from utilities import *
from imresize import imresize, imresize_to_shape
from architectures.VGG import VGG

class Baseline:
    def __init__(self, Generator, opt):
        self.Generator = Generator
        self.opt = opt

        self.Gs = []        # Generator list for each scale
        self.Zs = []        # Optimal noise list for each scale [z*, 0, 0, ..., 0]
        self.NoiseAmp = []  # Ratio of noise when merging with the output of the previous layer for each scale
        self.in_s = 0       # 0 Tensor with the downsampled dimensions of the input image for scale 0

        self.real_ = read_image(self.opt)
        self.style_ = read_image(self.opt, style=True)

        if self.style_.shape != self.real_.shape:
            self.style_ = imresize_to_shape(self.style_, [self.real_.shape[2], self.real_.shape[3]], opt)
            self.style_ = self.style_[:, :, :self.real_.shape[2], :self.real_.shape[3]]

        assert self.real_.shape == self.style_.shape
        dir2save = generate_dir2save(self.opt)
        if (os.path.exists(dir2save)):
            print("Would you look at that, the TrainedModel directory already exists!")
        else:
            try:
                os.makedirs(dir2save)
            except OSError:
                print("Making the directory really didn't work out, hyelp")

    def init_models(self):
        # Generator initialization
        netG = self.Generator(self.opt).to(self.opt.device)
        netG.apply(self.Generator.initialize_weight)
        if self.opt.netG != '':
            netG.load_state_dict(torch.load(self.opt.netG))
        print(netG)
        return netG

    def train(self):
        scale_num = 0
        nfc_prev = 0

        ### Load the VGG network
        vgg = VGG()
        vgg.load_state_dict(torch.load(self.opt.pretrained_VGG, map_location=self.opt.device))
        self.vgg = vgg.to(self.opt.device)
        # Make sure this network is frozen
        for parameter in self.vgg.parameters():
            parameter.requires_grad_(False)

        # Create output directory and save the downsampled image
        self.opt.out_ = generate_dir2save(self.opt)
        self.opt.outf = '%s/%d' % (self.opt.out_, scale_num)
        try:
            os.makedirs(self.opt.outf)
        except OSError:
            pass
        plt.imsave('%s/real_scale.png' % (self.opt.outf), convert_image_np(self.real_), vmin=0, vmax=1)
        netG = self.init_models()
        real = self.real_
        style = self.style_
        opt = self.opt

        opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
        opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

        # Pad the noise and image
        m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))

        # Generate z_opt template
        fixed_noise = generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
        z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
        z_opt = m_noise(z_opt)

        # Setup optimizer
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999), amsgrad=True)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

        # Store plots
        errG2plot = []
        errD2plot = []
        D_real2plot = []
        D_fake2plot = []
        z_opt2plot = []
        content_loss2plot = []
        style_loss2plot = []
        total_loss2plot = []

        # Training loop of a single scale
        for epoch in range(opt.niter):

            z_opt = generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            noise_ = generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
            # Multiple steps for G
            for j in range(opt.Gsteps):
                netG.zero_grad()

                # Only in the very first step of the very first epoch for a layer
                if (j == 0) and (epoch == 0):
                    # Define image and noise from previous scales (Nothing for Scale 0)
                    prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    self.in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                else:
                    prev = self.in_s
                    prev = m_image(prev)

                noise = noise_
                m_style = m_image(style)

                # Generate image with G and calculate loss from fake image
                fake = netG(real.detach(), prev, style.detach())

                # Reconstruction Loss
                content_loss = perceptual_loss(real, fake, self.vgg, opt.content_layers)

                # Style loss, layers from AdaIN
                style_loss = adain_style_loss(style, fake, self.vgg, opt.style_layers) * opt.alpha

                total_loss = content_loss + style_loss
                total_loss.backward()

                optimizerG.step()

            if epoch % opt.niter_update == 0 or epoch == (opt.niter - 1):
                plt.imsave('%s/fake_training.png' % (opt.outf), convert_image_np(fake.detach()), vmin=0, vmax=1)

                print("rec loss:", content_loss.item())
                print("style loss: ", style_loss.item())
                print("total loss: ", total_loss.item())


            if epoch % opt.niter_print == 0 or epoch == (opt.niter-1):
                print('Scale %d: Epoch [%d/%d]' % (len(self.Gs), epoch, opt.niter))
            content_loss2plot.append(content_loss)
            style_loss2plot.append(style_loss)
            total_loss2plot.append(total_loss)
            schedulerG.step()

        ### Create and Store Plots
        plt.figure(figsize=(12,9))
        plt.plot(content_loss2plot, label='Reconstruction Loss')
        plt.plot(style_loss2plot, label='Style Loss')
        plt.plot(total_loss2plot, label='Total Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, opt.niter)
        plt.legend()
        plt.grid(True)
        plt.savefig('%s/loss_plots.png' %  (opt.outf), bbox_inches='tight')


        torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
        torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))




