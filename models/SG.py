import torch.nn as nn
import torch

class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        self.resnet_DE = resnet_d1_e1()

    def forward(self, input_1):
        db= self.resnet_DE(input_1)
        return db

class resnet_d1_e1(nn.Module):
    def __init__(self):
        super(resnet_d1_e1, self).__init__()

        self.base1 = Base1()  #共享特征，VGG16前两个卷积block
        self.base2 = BaseDBD()  # 提取dbd特征；

    def forward(self, input_1):

        s1, s2, s3, s4 = self.base1(input_1)
        dbd1 = self.base2(s1, s2, s3, s4)

        return dbd1

class Base1(nn.Module):
    def __init__(self):
        super(Base1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1_1_2 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1_2 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1_2 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1_2 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, x):
        # print(x.size())
        x = self.conv1_1_2(x)
        x = self.conv1_2_2(x)
        x = self.maxpool(x)
        x = self.conv2_1_2(x)
        x = self.conv2_2_2(x)
        s1 = x
        x = self.maxpool(x)
        x = self.conv3_1_2(x)
        x = self.conv3_2_2(x)
        x = self.conv3_3_2(x)
        s2 = x

        x = self.maxpool(x)
        x = self.conv4_1_2(x)
        x = self.conv4_2_2(x)
        x = self.conv4_3_2(x)
        s3 = x
        # print(x.size())
        x = self.maxpool(x)
        x = self.conv5_1_2(x)
        x = self.conv5_2_2(x)
        x = self.conv5_3_2(x)
        s4 = x

        return s1, s2, s3, s4

class BaseDBD(nn.Module):
    def __init__(self):
        super(BaseDBD, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv3_1_2 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1_2 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv1_2 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        ##FFF1
        self.conv_fff_1_1 = BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_3 = BaseConv(256, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_1_4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_5 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_6 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)


        ##FFF2
        self.conv_fff_2_1 = BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_3 = BaseConv(128, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_2_4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_5 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_6 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)


        ##FFF3
        self.conv_fff_3_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_3 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_3_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_5 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)


        ##FFF4
        self.conv_fff_4_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_3 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_4_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_5 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        # for p in self.parameters():
        #     p.requires_grad = False
    def forward(self, s1, s2, s3, s4):

        x = s4
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s3], 1)
        x = self.conv1_2(x)
        x = self.conv2_2(x)


        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)


        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)


        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        x = self.conv_out_base_3(x)

        return x


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input



##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self,opt=None):
        super(Discriminator,self).__init__()
        self.ngpu = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),


            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data,torch.cuda.FloatTensor) and self.ngpu>1:
            output = nn.parallel.data_parallel(self.net,x,range(self.ngpu))
        else:
            output = self.net(x)
        return output

