# ------------------------------------------------------------------------------
# Using command line this file should be called with appropriate command line
# arguments, it will call the matching functions that will train or test the 
# given PyramidGAN. 
# ------------------------------------------------------------------------------


from parse import get_arguments
#from model3_vgg_training import PyramidGAN
from training import PyramidGAN
from architectures.first_architecture import Generator, Discriminator


if __name__ == "__main__":
    opt = get_arguments()

    sinGAN = PyramidGAN(Generator, Discriminator, opt)

    if opt.mode == "train":
        sinGAN.train()
    elif opt.mode == "random_samples" or "random_samples_arbitrary_sizes":
        sinGAN.generate_images()