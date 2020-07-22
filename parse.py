# ------------------------------------------------------------------------------
# Reads in the arguments from the command line and sets default values where 
# necessary.
# ------------------------------------------------------------------------------


import argparse
import torch
import random
from visdom import Visdom


def get_arguments():
    parser = argparse.ArgumentParser()

    # IO Parameters
    parser.add_argument('--content', help='Content input name', required=True)
    parser.add_argument('--test_content', help='Test content input name (replacing trained model content)', default=None)
    parser.add_argument('--content_dir', help='Content input directory', default='Images/Content')
    parser.add_argument('--style', help='Style input name', required=True)
    parser.add_argument('--test_style', help='Test style input name (replacing trained model style)', default=None)
    parser.add_argument('--style_dir', help='Style input directory', default='Images/Style')
    parser.add_argument('--out', help='Output folder', default='Output')
    parser.add_argument('--netG', help="Path to netG (to continue training)", default='')
    parser.add_argument('--netD', help="Path to netD (to continue training)", default='')

    parser.add_argument('--vis', help="Enable visualization through visdom", action="store_true")

    parser.add_argument('--pretrained_VGG', help="Relative path to pretrained VGG network file", default="VGG_pretrained.pth")
    

    # Workspace Parameters
    parser.add_argument('--mode', help='Task for network', default='train')
    parser.add_argument('--not_cuda', action='store_true', help='Disables cuda', default=0)    
    parser.add_argument('--manualSeed', type=int, help='Seed for randomizers')
    parser.add_argument('--nc_z', type=int, help='Noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='Image # channels', default=3)
        
    
    # Network Parameters
    parser.add_argument('--nfc', type=int, help='Number of feature channels for convolution', default=32)
    parser.add_argument('--min_nfc', type=int, help='Minimum number of feature channels for convolution', default=32)
    parser.add_argument('--ker_size', type=int, help='Kernel size of convolutions', default=3)
    parser.add_argument('--num_layer', type=int, help='Number of convolution layers', default=5)
    parser.add_argument('--stride', help='Stride of convolutions', default=1)
    # Padding should indeed be floor(kernel_size/2) so image size doesn't change during convolutions, but padding is added separately (outside of convolutions) to compensate.
    parser.add_argument('--padd_size', type=int, help='Convolution padding size', default=0) #math.floor(opt.ker_size/2)
    

    # Pyramid Parameters
    parser.add_argument('--scale_factor', type=float, help='Pyramid scale factor', default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp', type=float, help='Noise amplification weight', default=0.01)
    parser.add_argument('--min_size', type=int, help='Image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int, help='Image maximal size at the finest scale', default=250)


    # Training Parameters
    parser.add_argument('--niter', type=int, default=2000, help='Number of epochs to train per scale')
    parser.add_argument('--niter_print', type=int, default=25, help='How often to print progress during training')
    parser.add_argument('--niter_update', type=int, default=50, help='How often to update fake_sample image and G(z_opt) reconstructed image')
    parser.add_argument('--gamma', type=float, help='Scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='Learning rate for generator, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='Learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam, default=0.5')
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    parser.add_argument('--lambda_grad', type=float, help='Gradient penalty weight', default=0.1)

    parser.add_argument('--rec_weight', type=float, help='Reconstruction loss weight', default=10)
    parser.add_argument('--alpha', type=float, help='Style loss weight', default=1)

    parser.add_argument('--content_layers', nargs='+', help='VGG layers to calculate the content loss', default=['relu4_2'])
    parser.add_argument('--style_layers', nargs='+', help='VGG layers to calculate the style loss', default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])


    # Random Generation Parameters
    parser.add_argument('--gen_start_scale', type=int, help='Generation start scale', default=0)
    parser.add_argument('--gen_all_scales', help='Generate on all scales', action="store_true")
    parser.add_argument('--scale_h', type=float, help='Horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='Vertical resize factor for random samples', default=1)
    parser.add_argument('--swap_style', help='Name of trained model of which the style layers will be used', default=None)


    arguments = post_config(parser.parse_args())
    return arguments


def post_config(opt):
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")

    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.content[:-4], opt.scale_factor)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if (opt.vis):
        opt.vis = Visdom()

    return opt