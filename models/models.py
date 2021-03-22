import torch.nn as nn
import torch
from models.SG import GeneratorUNet,Discriminator
####################################################
# Initialize generator and discriminator
####################################################
def Create_nets(args):
    generator = GeneratorUNet()
    discriminator = Discriminator(args)
    discriminator2=Discriminator(args)

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        discriminator2=discriminator2.cuda()
    if args.epoch_start != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load('log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))
        discriminator.load_state_dict(torch.load('log/%s-%s/%s/discriminator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))
        discriminator2.load_state_dict(torch.load('log/%s-%s/%s/discriminator2_%d.pth' % (
        args.exp_name, args.dataset_name, args.model_result_dir, args.epoch_start)))

    return generator, discriminator,discriminator2

if __name__ == '__main__':
    c=Discriminator()
    print()