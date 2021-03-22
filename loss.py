#mask损失
import torch
import torch
import torch.nn.functional as F
from functools import partial
from models.vgg16 import VGG16

def min_permask_loss(mask, min_mask_coverage):
    '''
    One object mask per channel in this case
    '''
    return F.relu(min_mask_coverage - mask.mean(dim=(2, 3))).mean()

def binarization_loss(mask):
    return torch.min(1-mask, mask).mean()


class MaskLoss:
    def __init__(self, min_mask_coverage, mask_alpha, bin_alpha, min_mask_fn=min_permask_loss):
        self.min_mask_coverage = min_mask_coverage
        self.mask_alpha = mask_alpha
        self.bin_alpha = bin_alpha
        self.min_mask_fn = partial(min_mask_fn, min_mask_coverage=min_mask_coverage)

    def __call__(self, mask):
        if type(mask) in (tuple, list):
            mask = torch.cat(mask, dim=1)
        min_loss = self.min_mask_fn(mask)
        bin_loss = binarization_loss(mask)
        # return self.mask_alpha * min_loss + self.bin_alpha * bin_loss, dict(min_mask_loss=min_loss, bin_loss=bin_loss)
        # return self.mask_alpha * min_loss, dict(min_mask_loss=min_loss, bin_loss=bin_loss)
        return self.mask_alpha * min_loss + self.bin_alpha * bin_loss, dict(min_mask_loss=min_loss, bin_loss=bin_loss)
#重构损失
def Loss1(rendered,real):
    loss_fn = torch.nn.L1Loss(reduce=True,size_average=True)
    loss = loss_fn(rendered,real)
    return loss

def Get_loss_func(args):
    device = torch.device("cuda:0")
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    if torch.cuda.is_available():

        # criterion_GAN.cuda()
        # criterion_pixelwise.cuda()
        criterion_GAN.to(device)
        criterion_pixelwise.to(device)
    return criterion_GAN, criterion_pixelwise

def feather_loss(syn_celar,clear,syn_blur,blur):
    cuda_avail = torch.cuda.is_available()
    model = VGG16()
    for p in model.parameters():
        p.requires_grad = False

    if cuda_avail:
        model.cuda()
    # load model
    path = 'models/VGG16model_99'
    model_dict = torch.load(path)
    model.load_state_dict(model_dict['model'])
    model.eval()

    feather_syn_c, output_c = model(syn_celar)
    feather_c,output_cc = model(clear)
    # print(feather_syn_c.size())
    similarity=torch.cosine_similarity(feather_c,feather_syn_c,dim=1)
    loss_c=torch.mean(1-similarity,dim=0)
    # loss_c=l1loss(feather_syn_c,feather_c)

    feather_syn_b, output_b = model(syn_blur)
    feather_b, output_bb = model(blur)

    similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
    loss_b = torch.mean(1 - similarity2, dim=0)
    # loss_b=l1loss(feather_syn_b,feather_b)

    _, predictionb = torch.max(output_b.data, 1)#0
    _, predictionbb = torch.max(output_bb.data, 1)  #
    _, predictionc = torch.max(output_c.data, 1)#2
    nb=0
    for i in range(0,len(predictionb)):
        if predictionb[i]==0:
            nb+=1
    nc = 0
    for i in range(0, len(predictionc)):
        if predictionc[i] == 2:
            nc += 1
    return loss_c,loss_b,nc,nb