import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, n_classes, ndf = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(n_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)         # output => (N, 1, H/32, W/32)
        x = F.interpolate(x, size=(256, 320), mode='bilinear', align_corners=True)    # shape => (N, 1, H, W)
        x = F.sigmoid(x)

        return x

