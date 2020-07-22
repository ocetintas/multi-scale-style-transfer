# ------------------------------------------------------------------------------
# Utility functions that are used by other files/used by functions in this file.
# ------------------------------------------------------------------------------


import math
import torch
import os

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from skimage import io as img
from skimage import color

from imresize import imresize, norm, denorm, move_to_gpu, move_to_cpu, np2torch


# Function grouping scheme: functions that are called from other files have a 
# docstring, others are only used within this file. These local functions 
# usually follow directly on the function that calls them (whenever possible).



def generate_dir2save(opt):
    """
    Generate directory to save depending on the mode.

    Arguments:
        opt (argparse.ArgumentParser) : Command line arguments.

    Returns:
        dir2save (String) : Relative path to directory in which files are stored.
    """
    dir2save = None
    if (opt.mode == 'train'):
        dir2save = 'TrainedModels/%s/%s,scale_factor=%f,alpha=%f' % (opt.content[:-4], opt.style[:-4], opt.scale_factor_init, opt.alpha)

    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.content[:-4], opt.gen_start_scale)

    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrarySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.content[:-4], opt.scale_v, opt.scale_h)

    return dir2save
    

def read_image(opt, style=False):
    """
    Read input image in the specified path.

    Arguments:
        opt (argparse.ArgumentParser) : Command line arguments.
        style (bool) : When true, read_image loads in the style image, else it loads the content image.

    Returns:
        x (torch.cuda.FloatTensor) : Input image 
    """
    if style:
        x = img.imread('%s/%s' % (opt.style_dir, opt.style))
    else:
        x = img.imread('%s/%s' % (opt.content_dir, opt.content))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]

    return x


def adjust_scales2image(real_, opt):
    """
    Adjust scales of the pyramid according to the input image dimensions by modifying the "opt" parameters. Number of scales, scale 0 input dimension is decided in this function.

    Arguments:
        real_ (torch.cuda.FloatTensor) : Original image
        opt (argparse.ArgumentParser) : Command line arguments.

    Returns:
        real (torch.cuda.FloatTensor) : Image shape adjusted to the 1st scale
    
    Modifies input "opt"
    """
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1

    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)

    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop

    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size / min(real.shape[2], real.shape[3]), 1 / opt.stop_scale)
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop

    return real


def create_reals_pyramid(real, opt):
    """
    Creates downsampled versions of the training image for each scale.

    Arguments:
        real (torch.cuda.FloatTensor) : Input image.
        opt (argparse.ArgumentParser) : Command line arguments.

    Returns:
        reals (list) : Downscaled real image list for each scale
    """
    reals = []
    real = real[:, 0:3, :, :]
    for i in range(0, opt.stop_scale+1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale-i)
        curr_real = imresize(real, scale, opt)
        reals.append(curr_real)
    return reals


def convert_image_np(inp):
    """
    Converts torch image to numpy.
    """
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))

    inp = np.clip(inp,0,1)
    return inp


def reset_grads(model, require_grad):
    """
    Stops (if require_grad == False) calculation for gradient for certain objects after their training is completed.
    """
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    """
    Generate noise to input a certain scale.
    """
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])

    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2

    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)

    return noise

def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)


def draw_concat(Gs, Zs, reals, styles, NoiseAmp, in_s, mode, m_noise, m_image, opt):
    """
    Generates image from the lower scales for the input of current scale. Has two modes:
    rand: Generates images using new noise for each scale, and passing through all of them sequentially
    rec: Generates images deterministically from optimal noise list by using all lower scales.
    """
    G_z = in_s  # Input of the 0th scale which is a zero matrix
    if len(Gs) > 0:
        # Random image generation with new noise samples
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)

            # For each scale
            for G, Z_opt, real_curr, real_next, style_curr, noise_amp in zip(Gs, Zs, reals, reals[1:], styles, NoiseAmp):
                # Create noise
                if count == 0:
                    z = generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                style_curr = m_image(style_curr)
                z_in = noise_amp * z + G_z  # Weighted addition of the input image and noise
                G_z = G(z_in.detach(), G_z, style_curr.detach())   # Forward pass
                G_z = imresize(G_z,1/opt.scale_factor,opt)  # Upsample the image for the next scale
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        # Deterministic image generation with optimal noise
        elif mode == 'rec':
            count = 0
            # For each scale
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z  # Weighted addition of the input image and noise
                m_real_curr = m_image(real_curr)
                G_z = G(z_in.detach(),G_z, m_real_curr.detach())  # Forward pass
                G_z = imresize(G_z,1/opt.scale_factor,opt)  # Upsample the image for the next scale
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1

        elif mode == 'rec_from_style':
            count = 0
            # For each scale
            for G, Z_opt, real_curr, real_next, style_curr, noise_amp in zip(Gs, Zs, reals, reals[1:], styles, NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp * Z_opt + G_z  # Weighted addition of the input image and noise
                m_style_curr = m_image(style_curr)
                G_z = G(z_in.detach(), G_z, m_style_curr.detach())  # Forward pass
                G_z = imresize(G_z, 1 / opt.scale_factor, opt)  # Upsample the image for the next scale
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                # if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1

    return G_z


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    """
    Gradient Penalty for WGAN-GP Loss.
    """
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def load_trained_pyramid(opt):
    """
    Load pretrained pyramid network for image generation.
    """
    ### This way the folder name is defined in one function: generate_dir2save (changing it there will change it everywhere else).
    mode = opt.mode
    opt.mode = 'train'
    dir = generate_dir2save(opt)
    opt.mode = mode

    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        styles = torch.load('%s/styles.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('No appropriate trained model is exist, please train first.')

    return Gs, Zs, reals, styles, NoiseAmp


def generate_in2coarsest(reals, scale_v, scale_h, opt):
    """
    Calculate the zero input array for the generation start scale.

    Arguments:
        reals (list[torch.cuda.FloatTensor]) : Original image list
        scale_v (float) : Vertical scale
        scale_h (float) : Horizontal scale

    Returns:
        in_s (torch.cuda.FloatTensor) : Input image shape adjusted to the generation scale
    """
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s


def SinGAN_generate(Gs, Zs, reals, styles, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=10):
    """
    Generate image with the given parameters.
    Returns:
        I_curr(torch.cuda.FloatTensor) : Current Image
    """
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G, Z_opt, noise_amp, style in zip(Gs, Zs, NoiseAmp, styles):
        pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2] - pad1*2) * scale_v
        nzy = (Z_opt.shape[3] - pad1*2) * scale_h

        images_prev = images_cur
        images_cur = []
        m_style = m(style)

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)

                I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                I_prev = m(I_prev)
                I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                I_prev = upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev, m_style.detach())

            if n == len(reals)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.content[:-4], gen_start_scale)
                else:
                    dir2save = generate_dir2save(opt)

                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass

                plt.imsave('%s/%d.png' % (dir2save, i), convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)

            images_cur.append(I_curr)
        n += 1
    return I_curr.detach()



def mean_std(feature_space, eps=1e-5):
    """
    Calculates the mean and standard deviation for each channel

    Arguments:
        feature_space (torch.Tensor): Feature space of shape (N, C, H, W)
    """
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feature_space.size()
    assert (len(size) == 4), "Feature space shape is NOT structured as N, C, H, W!"
    N, C = size[:2]
    feat_var = feature_space.view(N, C, -1).var(dim=2) + eps
    feature_std = feat_var.sqrt().view(N, C, 1, 1)
    feature_mean = feature_space.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feature_mean, feature_std


def adain(content_feat, style_feat):
    """
    Adaptive instance normalization layer. Takes content and style features and returns content features with changed
    mean and variance according to AdaIN formula.

    Arguments:
        content_feat (torch.Tensor): Content feature space of shape (N, C, H, W)
        style_feat (torch.Tensor): Style feature space of shape (N, C, H, W)
    """
    assert (content_feat.size()[:2] == style_feat.size()[:2]), "Content and feature space N or C does NOT match!"
    size = content_feat.size()
    style_mean, style_std = mean_std(style_feat)
    content_mean, content_std = mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)



def swap_style_weights (content_generator, style_generator):
    """
    Swaps the weights in both generators for all layers that involve the style input. Make sure the style layer variables in the generator have the word 'style' in their name, e.g. style_head, style_body, style_tail.
    Both generators should be trained on the same architecture such that the names match!

    Arguments:
        content_generator (list of torch.nn.Module) : Generator models of each scale that should receive new style weights

        style_generator (list of torch.nn.Module) : Generator models of each scale that should contain new style weights that will be transferred over to the content_generator

    Returns:
        result_generator (list of torch.nn.Module) : Generator models of each scale that will contain the content weights from content_generator and the style weights from style_generator
    """
    result_generator = []

    for scale in range(len(content_generator)):
        G_curr = content_generator[scale]
        sd = G_curr.state_dict()

        for layer1, layer2 in zip(G_curr.state_dict().items(), style_generator[scale].state_dict().items()):
            # Check if the layer has anything to do with the style
            if layer1[0].find('style') != -1:
                # Check if both layers have the same name
                if layer1[0] == layer2[0]:
                    sd[layer1[0]] = layer2[1]
                else:
                    print("Mismatch of style layer %s" % layer1[0])
        
        G_curr.load_state_dict(sd)
        result_generator.append(G_curr)

    return result_generator


def perceptual_loss(real_image, fake_image, vgg_network, layers):
    """
    Calculates the perceptual loss according to features from given layers of a pretrained vgg_network. 
    The two inputs real_image and fake_image are both passed to the VGG network until (and including) the given layer (or list of layers) and the resulting features are compared using MSE loss.

    Arguments:
        layers (list of Strings) : Names of VGG layers. The output of the VGG network at that layer (or those layers) will be used to calculate the loss.

    Returns:
        p_loss (torch.cuda.FloatTensor) : Perceptual loss of given inputs
    """
    assert (real_image.size() == fake_image.size()), "Size mismatch between real and fake image!"

    real_features = vgg_network(real_image, layers)
    fake_features = vgg_network(fake_image, layers)

    loss = nn.MSELoss()
    p_loss = 0
    for l in range(len(layers)):
        p_loss += loss(real_features[l], fake_features[l])

    # Normalize with number of layers (assume same weight)
    p_loss = p_loss / len(layers)

    return p_loss


def adain_style_loss(style, fake_image, vgg_network, layers):
    """
    Perceptual loss on style similar to the version implemented in AdaIN, using mean and standard deviation.
    """
    assert (style.size() == fake_image.size()), "Size mismatch between style and fake image!"

    style_features = vgg_network(style, layers)
    fake_features = vgg_network(fake_image, layers)

    loss = nn.MSELoss()
    L_s = 0
    temp_weights = [1/0.5, 1/5, 1/30, 1/60]
    for l in range(len(layers)):
        mean_fake, std_fake = mean_std(fake_features[l])
        mean_style, std_style = mean_std(style_features[l])

        L_s +=  (loss(mean_style, mean_fake) + loss(std_style, std_fake))

    # Normalize with number of layers (assume same weight)
    L_s = L_s / len(layers)

    return L_s













