import torch.nn as nn
import torch.nn.functional as F



class DeconvBn_2(nn.Module):
    """ Deconvolution(stride=2) => Batch Normilization """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        return self.bn(self.deconv(x))



class DeconvBn_8(nn.Module):
    """ Deconvolution(stride=8) => Batch Normilization """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=8, stride=8, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        return self.bn(self.deconv(x))


class FCDiscriminator(nn.Module):

    def __init__(self, config, ndf = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(self.config.n_classes + self.config.in_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.deconv_bn1 = DeconvBn_2(512, 512)
        self.deconv_bn2 = DeconvBn_2(512, 256)
        self.deconv_bn3 = DeconvBn_8(256, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x3 = self.conv3(x)
        x3 = self.leaky_relu(x3)
        x4 = self.conv4(x3)
        x4 = self.leaky_relu(x4)
        x5 = self.conv5(x4)         # output => (N, 1, H/32, W/32)
        
        feature = self.deconv_bn1(x5)
        feature = self.deconv_bn2(x4 + feature)
        feature = x3 + feature
        out = self.deconv_bn3(feature)
        out = F.sigmoid(out)

        if self.config.feature_match:
            return feature, out
        else:
            return out
