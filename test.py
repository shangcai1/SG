import argparse
import os
import torch
from tqdm import tqdm
from utils import to_image_test
from torch.autograd import Variable
from datasets import Get_dataloader_test
from models.SG import GeneratorUNet
device = torch.device("cuda:0")

def test(stict,mask_save_path,image_path):

    if not torch.cuda.is_available():
        generator2 = GeneratorUNet()
    else:
        generator2 = GeneratorUNet().cuda()
    generator2.load_state_dict(torch.load(stict))
    generator2.eval()

    dataloder = Get_dataloader_test(image_path, 1)
    for i,(img,index) in tqdm(enumerate(dataloder)):
        if not torch.cuda.is_available():
            img=Variable(img)
        else:
            img = Variable(img).cuda()
        mask = generator2(img)

        os.makedirs(mask_save_path, exist_ok=True)
        to_image_test(mask, i=int(index.data.numpy()),tag='', path=mask_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--stict', default='./checkpoints/generator_23000.pth',type=str)
    parser.add_argument('--image_path', default='./dataset/test/dut500-source', type=str)
    parser.add_argument('--mask_save_path', default='./results/test_dut', type=str)
    args=parser.parse_args()
    #test
    test(args.stict,args.mask_save_path,args.image_path)




























