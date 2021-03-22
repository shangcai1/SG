import torch
# Optimizers
def Get_optimizers(args, generator, discriminator,discriminator2):

    optimizer_G = torch.optim.SGD(
        generator.parameters(),
        lr=args.lr, momentum=0.5)
    optimizer_D = torch.optim.SGD(
        discriminator.parameters(),
        lr=args.lr, momentum=0.5)
    optimizer_D2 = torch.optim.SGD(
        discriminator2.parameters(),
        lr=args.lr,  momentum=0.5)

    return optimizer_G, optimizer_D,optimizer_D2

# Loss functions
def Get_loss_func(args):
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.MSELoss()
    if torch.cuda.is_available():
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
    return criterion_GAN, criterion_pixelwise

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
