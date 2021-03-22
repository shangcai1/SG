import argparse
import os
import numpy as np
import torch
from PIL import Image
from sklearn import metrics

def eval1(mask_path,gt_path,m):
    files=os.listdir(gt_path)
    maes=0
    precesions=0
    recalls=0
    fmeasures=0
    for file in files:
        mask1=mask_path+'/'+file
        gt1=gt_path+'/'+file
        # mask=np.array(Image.open(mask1))
        mask1 = Image.open(mask1)
        mask1 = mask1.resize((320, 320))
        mask = np.array(mask1)
        mask = mask.astype(float)/255.0
        mask_1 = mask

        (w,h)=mask.shape
        zeros = np.zeros((w, h))
        if m>1:
            mean=np.mean(mask)*1.5
        else:
            mean=m
        if mean>1:
            mean=1
        for i in range(w):
            for j in range(h):
                if mask_1[i,j]>=mean:
                    zeros[i,j]=1.0
                else:
                    zeros[i,j]=0.0

        gt=(np.array(Image.open(gt1)).astype(float))/255.0
        for i in range(w):
            for j in range(h):
                if gt[i, j] > 0.1:
                    gt[i, j] = 1.0
                else:
                    gt[i, j] = 0.0

        mae=np.mean(np.abs((gt-mask)))
        maes+=mae
        precesion=metrics.precision_score(gt.reshape(-1), zeros.reshape(-1))
        precesions+=precesion
        recall=metrics.recall_score(gt.reshape(-1), zeros.reshape(-1))
        recalls+=recall
        if precesion==0 and recall==0:
            fmeasure=0.0
        else:
            fmeasure=((1+0.3)*precesion*recall)/(0.3*precesion+recall)
        fmeasures+=fmeasure
    mae1=maes/len(files)
    fmeasure1=fmeasures/len(files)
    recall1=recalls/len(files)
    precesion1=precesions/len(files)
    return mae1,fmeasure1,recall1,precesion1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--mask_path', default='./results/test_dut', type=str)
    parser.add_argument('--gt_path', default='./dataset/test/dut500-gt', type=str)
    args=parser.parse_args()

    #eval
    mae, fm, r, p = eval1(args.mask_path, args.gt_path, 1.5)
    print('mae:%.3f,fm:%.3f' % (mae, fm))