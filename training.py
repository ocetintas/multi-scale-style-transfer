# ------------------------------------------------------------------------------
# Contains the class PyramidGAN which is used for training and testing 
# generator/discriminator architectures in pyramidal structure.
# ------------------------------------------------------------------------------


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
import numpy as np

# Training model
from utilities import generate_dir2save, read_image, adjust_scales2image, create_reals_pyramid, convert_image_np, reset_grads, generate_noise, draw_concat, calc_gradient_penalty
from imresize import imresize

# Generating images
from utilities import load_trained_pyramid, generate_in2coarsest, SinGAN_generate



class PyramidGAN:
    """
    Defines a class containing all necessary information and functions to train a given GAN on multiple scales.

    Arguments for initialization:
        Generator (nn.Module) : The generator architecture class that will be used to create generator models.
        Discriminator (nn.Module) : The discriminator architecture class that will be used to create discriminator models.
        opt (ArgumentParser) : Parsed input arguments.
    """
    def __init__(self, Generator, Discriminator, opt):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.opt = opt

        ### Set parameters for the training of the 0th layer
        self.Gs = []        # Generator list for each scale
        self.Zs = []        # Optimal noise list for each scale [z*, 0, 0, ..., 0]
        self.NoiseAmp = []  # Ratio of noise when merging with the output of the previous layer for each scale
        self.in_s = 0       # 0 Tensor with the downsampled dimensions of the input image for scale 0

        ### Content image pyramid
        self.real_ = read_image(self.opt)
        # "adjust_scales2image" also arranges network parameters according to input dimensions
        self.real = adjust_scales2image(self.real_, self.opt)     
        self.reals = create_reals_pyramid(self.real, self.opt)

        # For visualization, let's just for now do maximal image dimensions
        self.opt.viswindows = []    # Windows in visdom that is updated during training G(z_opt)
        self.max_width = convert_image_np(self.real).shape[0]
        self.max_height = convert_image_np(self.real).shape[1]

        ### Style image pyramid
        if (opt.style_input is not None):
            self.style_ = read_image(self.opt, style=True)
            self.style = imresize(self.style_, self.opt.scale1, self.opt)      
            self.styles = create_reals_pyramid(self.style, self.opt)

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

        # Discriminator initialization
        netD = self.Discriminator(self.opt).to(self.opt.device)
        netD.apply(self.Discriminator.initialize_weight)
        if self.opt.netD != '':
            netD.load_state_dict(torch.load(self.opt.netD))
        print(netD)

        return netD, netG


    def train(self):
        """
        Trains GAN for niter epochs over stop_scale number of scales. Main training loop that calls train_scale.
        Controls transition between layers. After training is done for a certain layer, freezes weights of the trained scale, and arranges computational graph by changing requires_grad parameters.
        """
        scale_num = 0
        nfc_prev = 0

        # Training loop for each scale
        while scale_num < self.opt.stop_scale+1:
            # Number of filters in D and G changes every 4th scale
            self.opt.nfc = min(self.opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            self.opt.min_nfc = min(self.opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            # Create output directory and save the downsampled image
            self.opt.out_ = generate_dir2save(self.opt)
            self.opt.outf = '%s/%d' % (self.opt.out_, scale_num)
            try:
                os.makedirs(self.opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (self.opt.out_), convert_image_np(self.real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (self.opt.out_), convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/real_scale.png' %  (self.opt.outf), convert_image_np(self.reals[scale_num]), vmin=0, vmax=1)

            # Initialize D and G of the current scale. D and G will be initialized with the previous scale's weights if the dimensions match.
            D_curr,G_curr = self.init_models()
            if (nfc_prev == self.opt.nfc):
                G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (self.opt.out_, scale_num-1)))
                D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (self.opt.out_, scale_num-1)))

            # Training of single scale
            z_curr, G_curr = self.train_scale(G_curr, D_curr, self.opt)

            # Stop gradient calculation for G and D of current scale
            G_curr = reset_grads(G_curr, False)
            G_curr.eval()
            D_curr = reset_grads(D_curr, False)
            D_curr.eval()

            # Store the necessary variables of this scale
            self.Gs.append(G_curr)
            self.Zs.append(z_curr)
            self.NoiseAmp.append(self.opt.noise_amp)

            # Save the networks and important parameters
            torch.save(self.Zs, '%s/Zs.pth' % (self.opt.out_))
            torch.save(self.Gs, '%s/Gs.pth' % (self.opt.out_))
            torch.save(self.reals, '%s/reals.pth' % (self.opt.out_))
            torch.save(self.NoiseAmp, '%s/NoiseAmp.pth' % (self.opt.out_))

            scale_num+=1
            nfc_prev = self.opt.nfc  # Update the number of filters
            del D_curr,G_curr

        return


    def train_scale(self, netG, netD, opt):
        """
        Main training loop of a SINGLE scale. Trains the generator and discriminator of a single scale. Trains G and D
        separately and takes care of everything related to forward and backward pass. Includes also the training plots.

        Arguments:
            netD (nn.Module) : Discriminator of the current level
            netG (nn.Module) : Generator of the current level
            opt (argparse.ArgumentParser) : Command line arguments.

        Returns:
            z_opt (torch.Tensor) : Optimal noise for current level
            netG (nn.Module) : Trained generator of current level

        Modifies input "opt"
        """

        real = self.reals[len(self.Gs)]
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
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

        # Store plots
        errG2plot = []
        errD2plot = []
        D_real2plot = []
        D_fake2plot = []
        z_opt2plot = []

        z_prev = None

        # Creating vis windows
        if (opt.vis != False):
            zopt_window = opt.vis.image(
                np.zeros_like(convert_image_np(real).transpose(2,0,1)), 
                opts=dict(
                    title='G(z_opt) on scale %d' % len(self.Gs),
                    width=self.max_width, 
                    height=self.max_height
                    )
                )
            real_window = opt.vis.image(
                convert_image_np(real).transpose(2,0,1), 
                opts=dict(
                    title='Real image on scale %d' % len(self.Gs), 
                    width=self.max_width, 
                    height=self.max_height
                    )
                )
            opt.viswindows.append(zopt_window)

        # Training loop of a single scale
        for epoch in range(opt.niter):
            # A little too many arguments... Maybe I should revert it back to original SinGAN style.
            errG, errD, D_real, D_fake, rec_loss, z_opt, z_prev = self.train_epoch(netG, netD, optimizerG, optimizerD, real, m_noise, m_image, epoch, z_opt, z_prev, opt)

            errG2plot.append(errG)
            errD2plot.append(errD)
            D_real2plot.append(D_real)
            D_fake2plot.append(D_fake)
            z_opt2plot.append(rec_loss)


            if epoch % opt.niter_print == 0 or epoch == (opt.niter-1):
                print('Scale %d: Epoch [%d/%d]' % (len(self.Gs), epoch, opt.niter))

                torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

            # Scheduler steps
            schedulerD.step()
            schedulerG.step()

        ### Create and Store Plots
        plt.figure(figsize=(12,9))

        plt.plot(errG2plot, label='Generator Error')
        plt.plot(errD2plot, label='Discriminator Error')
        plt.plot(D_real2plot, label='Real Image Discriminator Loss')
        plt.plot(D_fake2plot, label='Fake Image Discriminator Loss')
        plt.plot(z_opt2plot, label='Reconstruction Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, opt.niter)
        plt.legend()
        plt.grid(True)
        plt.savefig('%s/loss_plots.png' %  (opt.outf), bbox_inches='tight')


        torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
        torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
        torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        return z_opt, netG   


    # I noticed far too late that so many variables are needed from previous scales. Having to pull all of those variables out into the function arguments might not be worth it after all for clarity.
    def train_epoch(self, netG, netD, optimizerG, optimizerD, real, m_noise, m_image, epoch, z_opt, z_prev, opt):
        """
        Trains network for one epoch.

        Arguments:
            epoch (int) : Current epoch.
            z_prev () : Can be None on the first epoch.
            opt (argparse.ArgumentParser) : Command line arguments.

        Returns:
        TODO: Check if these are actually torch.cuda.Float types
            errG (torch.cuda.Float) : Error of generator
            errD (torch.cuda.Float) : Error of discriminator
            D_x (torch.cuda.Float) : Error of discriminator on original image
            D_G_z (torch.cuda.Float) : Error of discriminator on fake image
            rec_loss (torch.cuda.Float) : Reconstruction loss
            z_prev
        """
        # Scale 0
        if (self.Gs == []):
            # Generate optimal noise that will be kept fixed during training
            z_opt = generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            noise_ = generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
        else:
            noise_ = generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        # Multiple steps for D
        for j in range(opt.Dsteps):
            # TRAIN WITH REAL IMAGE
            netD.zero_grad()

            output = netD(real).to(opt.device)
            D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # TRAIN WITH FAKE IMAGE
            # Only in the very first step of the very first epoch for a layer
            if (j == 0) and (epoch == 0):
                # Scale 0
                if (self.Gs == []):
                    # Define image and noise from previous scales (Nothing for Scale 0)
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    self.in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                # Remaining scales other than 0
                else:
                    # Calculate image and noise from previous scales with draw_concat function
                    prev = draw_concat(self.Gs, self.Zs, self.reals, self.NoiseAmp, self.in_s, 'rand', m_noise, m_image, opt)    # Randomly generate image using previous scales
                    prev = m_image(prev)
                    z_prev = draw_concat(self.Gs, self.Zs, self.reals, self.NoiseAmp, self.in_s,'rec', m_noise, m_image, opt)   # Generate image with optimal noise using previous scales
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))  # noise amplitude for a certain layer is decided according to the performance of previous layers
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
            # If not very first epoch, just generate previous image
            else:
                prev = draw_concat(self.Gs, self.Zs, self.reals, self.NoiseAmp, self.in_s, 'rand', m_noise, m_image, opt)
                prev = m_image(prev)


            # Scale 0
            if (self.Gs == []):
                noise = noise_
            # Other scales
            else:
                noise = opt.noise_amp * noise_ + prev

            # Generate image with G and calculate loss from fake image
            fake = netG(noise.detach(),prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            # WGAN Loss with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        # Multiple steps for G
        for j in range(opt.Gsteps):
            netG.zero_grad()
            # Calculate loss
            output = netD(fake)
            D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            # Reconstruction Loss
            if opt.alpha != 0:
                loss = nn.MSELoss()
                ### No painting in our version 
                # if opt.mode == 'paint_train':
                #     z_prev = functions.quant2centers(z_prev, centers)
                #     plt.imsave('%s/z_prev.png' % (opt.outf), convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp * z_opt + z_prev
                rec_loss = opt.alpha * loss(netG(Z_opt.detach(), z_prev), real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        if epoch % opt.niter_update == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' % (opt.outf), convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png' % (opt.outf), convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            plt.imsave('%s/D_fake.png' % (opt.outf), convert_image_np(D_fake_map))
            plt.imsave('%s/D_real.png' % (opt.outf), convert_image_np(D_real_map))
            plt.imsave('%s/z_opt.png' % (opt.outf), convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            plt.imsave('%s/prev.png' % (opt.outf), convert_image_np(prev), vmin=0, vmax=1)
            plt.imsave('%s/noise.png' % (opt.outf), convert_image_np(noise), vmin=0, vmax=1)
            plt.imsave('%s/z_prev.png' % (opt.outf), convert_image_np(z_prev), vmin=0, vmax=1)

            if (opt.vis != False):
                opt.vis.image(
                    convert_image_np(netG(Z_opt.detach(), z_prev).detach()).transpose(2,0,1),
                    win=opt.viswindows[-1], 
                    opts=dict(
                        title='G(z_opt) on scale %d' % len(self.Gs), 
                        width=self.max_width, 
                        height=self.max_height
                        )
                    )

        return (errG.detach()+rec_loss), errD.detach(), D_x, D_G_z, rec_loss, z_opt, z_prev


    def generate_images(self):
        """
        Generates images using trained model.

        Returns:
            Creates directory for output images to be stored in.
        """
        opt = self.opt
        # TODO: Can the load_trained_pyramid be used to load in Gs, Zs, reals and NoisAmp into the class PyramidGAN and used in general for all cases aside from training?
        Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)

        if opt.mode == 'random_samples':
            in_s = generate_in2coarsest(reals, 1, 1, opt)
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)
        elif opt.mode == 'random_samples_arbitrary_sizes':
            in_s = generate_in2coarsest(reals, opt.scale_v, opt.scale_h, opt)
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h)

        return