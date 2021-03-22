from loss import *
from models.models import Create_nets
from datasets import *
from options import TrainOptions
from optimizer import *
from test import test
from eval import eval1
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"
#load the args
args = TrainOptions().parse()

# Initialize generator and discriminator
generator, discriminator,discriminator2 = Create_nets(args)
# Loss functions
criterion_GAN, criterion_pixelwise = Get_loss_func(args)
mask_loss_fn = MaskLoss(args.min_mask_coverage, args.mask_alpha, args.binarization_alpha)
# Optimizers
optimizer_G, optimizer_D,optimizer_D2 = Get_optimizers(args, generator, discriminator,discriminator2)
log={'bestmae_it':0,'best_mae':10,'fm':0,'bestfm_it':0,'best_fm':0,'mae':0}
# Configure dataloaders
real_loder = Get_dataloader(args.path_gt, args.batch_size)
dis_C_loder1 = Get_dataloader(args.path_clear, args.batch_size)
dis_C_loder2 = Get_dataloader(args.path_clear, args.batch_size)

dis_B_loder1 = Get_dataloader(args.path_blur, args.batch_size)
dis_B_loder2 = Get_dataloader(args.path_blur, args.batch_size)

real = iter(real_loder)
dis_C1 = iter(dis_C_loder1)
dis_C2 = iter(dis_C_loder2)

dis_B1 = iter(dis_B_loder1)
dis_B2 = iter(dis_B_loder2)

j=0
# 开始训练
pbar = range(args.epoch_start,3_000_000)
for i in pbar:
    try:
        real_image = next(real)
        real_image = real_image.to(device)

        dis_C_image1 = next(dis_C1)
        dis_C_image1 = dis_C_image1.to(device)
        dis_C_image2 = next(dis_C2)
        dis_C_image2 = dis_C_image2.to(device)

        dis_B_image1 = next(dis_B1)
        dis_B_image1 = dis_B_image1.to(device)
        dis_B_image2 = next(dis_B2)
        dis_B_image2 = dis_B_image2.to(device)

    except (OSError, StopIteration):
        real = iter(real_loder)
        dis_C1 = iter(dis_C_loder1)
        dis_C2 = iter(dis_C_loder2)

        dis_B1 = iter(dis_B_loder1)
        dis_B2 = iter(dis_B_loder2)

        real_image = next(real)
        real_image = real_image.to(device)

        dis_C_image1 = next(dis_C1)
        dis_C_image1 = dis_C_image1.to(device)
        dis_C_image2 = next(dis_C2)
        dis_C_image2 = dis_C_image2.to(device)

        dis_B_image1 = next(dis_B1)
        dis_B_image1 = dis_B_image1.to(device)
        dis_B_image2 = next(dis_B2)
        dis_B_image2 = dis_B_image2.to(device)


    # ------------------
    #  Train Generators
    # ------------------

    # Adversarial ground truths
    patch=(1,1,1)
    valid = Variable(torch.FloatTensor(np.ones((real_image.size(0),*patch))).cuda(), requires_grad=False)
    fake = Variable(torch.FloatTensor(np.zeros((real_image.size(0),*patch))).cuda(), requires_grad=False)

    optimizer_G.zero_grad()
    requires_grad(generator, True)
    requires_grad(discriminator, False)
    requires_grad(discriminator2, False)

    mask = generator(real_image)
    Mask1=mask
    # syn image
    syn_image_clear = Mask1 * real_image + (1 - Mask1) * dis_C_image1
    syn_image_blur = Mask1 * dis_B_image1 + (1 - Mask1) * real_image

    pred_fake = discriminator(syn_image_clear)
    loss_GAN1 = criterion_GAN(pred_fake, valid)
    pred_fake2 = discriminator2(syn_image_blur)
    loss_GAN2 = criterion_GAN(pred_fake2, valid)

    loss_mask, _ = mask_loss_fn(mask)
    loss_c,loss_b,c,b = feather_loss(syn_image_clear, dis_C_image2, syn_image_blur, dis_B_image2)

    # Total loss
    loss_G = 0.01*(loss_GAN1+loss_GAN2)+loss_mask+(loss_c+loss_b)
    # loss_G=loss_GAN2+loss_mask+loss_GAN1
    loss_G.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # # ---------------------
    optimizer_D.zero_grad()
    requires_grad(generator, False)
    requires_grad(discriminator, True)
    requires_grad(discriminator2, False)
    # Real loss
    pred_real = discriminator(dis_C_image2)
    loss_real = criterion_GAN(pred_real, valid)

    # Fake loss
    pred_fake = discriminator(syn_image_clear.detach())
    loss_fake = criterion_GAN(pred_fake, fake)
    # print(pred_fake)
    # print(loss_fake)

    # Total loss
    loss_D = 0.5 * (loss_real + loss_fake)
    loss_D.backward()
    optimizer_D.step()

    optimizer_D2.zero_grad()
    requires_grad(generator, False)
    requires_grad(discriminator, False)
    requires_grad(discriminator2, True)
    # Real loss
    pred_real2 = discriminator2(dis_B_image2)
    loss_real2 = criterion_GAN(pred_real2, valid)

    # Fake loss
    pred_fake2 = discriminator2(syn_image_blur.detach())
    loss_fake2 = criterion_GAN(pred_fake2, fake)

    # Total loss
    loss_D2 = 0.5 * (loss_real2 + loss_fake2)
    loss_D2.backward()
    optimizer_D2.step()

    if i%1000==0:
        print(
            "\r[Batch%d]-[Dloss:%f,Dloss2:%f]-[loss_mask:%f, loss_GAN1:%f,loss_GAN2:%f]" %
            (i,loss_D.data.cpu(),loss_D2.data.cpu(),
             loss_mask.data.cpu(), loss_GAN1.data.cpu(),loss_GAN2.data.cpu()))

    if i % 1000==0:
        image_path = 'log/%s-%s/%s' % (args.exp_name, args.dataset_name, args.img_result_dir)
        os.makedirs(image_path, exist_ok=True)
        to_image(real_image, i=i, tag='input', path=image_path)
        to_image(syn_image_clear, i=i, tag='syn_image', path=image_path)
        to_image(syn_image_blur, i=i, tag='syn_blur', path=image_path)
        to_image_mask(mask, i=i, tag='mask', path=image_path)

    if args.checkpoint_interval != -1 and i % 1000== 0:
    # Save model checkpoints
        torch.save(generator.state_dict(), 'log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,i))
        torch.save(discriminator.state_dict(), 'log/%s-%s/%s/discriminator1_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))
        torch.save(discriminator2.state_dict(), 'log/%s-%s/%s/discriminator2_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))

        pthpath='log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,i)
        mask_save_path = 'log/%s-%s/test/test100-%s' % (args.exp_name, args.dataset_name, i)
        image_path= './dataset/test/xu100-source'
        test(pthpath,mask_save_path,image_path)

        gt_path = './dataset/test/xu100-gt'
        mae1,fmeasure1,_,_=eval1(mask_save_path,gt_path,1.5)

        if mae1<log['best_mae'] :
            log['bestmae_it']=i
            log['best_mae']=mae1
            log['fm']=fmeasure1
        if fmeasure1>log['best_fm']:
            log['bestfm_it']=i
            log['best_fm']=fmeasure1
            log['mae']=mae1
        print('====================================================================================================================')
        print('batch:',i, "mae:", mae1, "fmeasure:", fmeasure1)
        print('bestmae_it',log['bestmae_it'],'best_mae',log['best_mae'],'fm:',log['fm'])
        print('bestfm_it',log['bestfm_it'],'mae:',log['mae'],'best_fm',log['best_fm'])
        print('=====================================================================================================================')