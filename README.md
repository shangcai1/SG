# SG in PyTorch

Implementation of "Self-generated Defocus Blur Detection via Dual Adversarial Discriminators" in PyTorch

## Datasets
- `train_data`: The data for training.
  - `1204source`: Contains 604 training images of CUHK Dataset and 600 training images of DUT Dataset, 1204 in total.
  - `FCFB`: Including 500 natural full clear images(FC) and 500 natural full blurred images(FB).
    - `FC`: 500 natural full clear images.
    - `FB`: 500 natural full blurred images.
- `test_data`: The data for testing.
  - `CUHK`: Contains 100 testing images of CUHK Dataset and it's GT.
  - `DUT`: Contains 500 testing images of DUT Dataset and it's GT.

baidu link: https://pan.baidu.com/s/1_3Z7w9IrlqOU25QbVdhytQ  passward: aktv

google link: https://drive.google.com/file/d/1DtWbMUppxa8eC0O3ZVBjLQO_i5GUG1pa/view?usp=sharing

### Test
You can use the following command to test：
> python test.py --stict PRETRAINED_WEIGHT --image_path IMG_PATH --mask_save_path SAVE_PATH

We have trained the model and We got FM about 0.701 and MAE about 0.172 on DUT Dataset and got FM about 0.769 and MAE about 0.119 on CUHK Dataset. You can use the following model to output results directly.

Here is our parameters：
baidu link: https://pan.baidu.com/s/1XbRiUnXr6LlgFNbWUWjQ3g  passward: nm2l

google link: https://drive.google.com/file/d/1Q6odk3yGexqFIt6Iu3WNFRoInWealy_2/view?usp=sharing

Put "generator.pth" in "./checkpoints".

## Train
You can use the following command to train：
> python train.py --path_clear CLEAT_DATA_PATH --path_blur BLUR_DATA_PATH --path_gt GT_DATA_PATH 

- `train.py`: the entry point for training.
- `models/SG.py`: defines the architecture of the Generator models and Discriminator models.
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well.
- `datasets/`: process the dataset before passing to the network.
- `models/vgg16.py`: defines the Classifier.
- `models/models.py`: defines the model.
- `optimizer.py`: defines the optimizetion.
- `loss.py`: defines the loss.

Here is the pretrained weight of the Classifier.Put "VGG16model" in "./models" 

baidu link: https://pan.baidu.com/s/1ZrtEVoYQjUVUWgzJ68gwvQ  passward: 26jm

google link: https://drive.google.com/file/d/1K4OAo-WEKmizFt4edQzeqBp-eNgurZft/view?usp=sharing

### Eval
If you want to use FM and MAE to evaluate the results, you can use the following code：

> python eval.py --mask_path MASK_PATH --gt_path GT_PATH

If you want to get the PR curve and FM curve in the article, use the following code:

> python plt_pr.py
> 
> python plt_fm.py

