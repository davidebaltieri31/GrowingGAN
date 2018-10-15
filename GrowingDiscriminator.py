import torch
import torch.nn as nn
from GrowingGenerator import NoiseLayer

class GrowingDiscriminator(nn.Module):
    def block(self, input_channels, output_channels, use_noise, use_tanh, use_groupnorm, longer):
        return nn.Sequential(
            NoiseLayer() if use_noise else nn.Sequential(),
            nn.Conv2d(input_channels, output_channels,kernel_size=4, stride=2,padding=1, bias=False),
            nn.GroupNorm(1, output_channels) if use_groupnorm else nn.BatchNorm2d(output_channels),
            nn.Tanh() if use_tanh else nn.LeakyReLU(0.1, True),
            nn.Sequential( NoiseLayer() if use_noise else nn.Sequential(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, output_channels)if use_groupnorm else nn.BatchNorm2d(output_channels),
            nn.Tanh() if use_tanh else nn.LeakyReLU(0.1, True)) if longer else nn.Sequential()
        )

    def __init__(self, device,
                 input_channels,
                 output_size,
                 starting_size=4,
                 use_layer_noise=False,
                 use_tanh = False,
                 use_groupnorm = False,
                 longer_network = False,
                 layer_channel=128):

        if(output_size==1):
            self.output_size = output_size
        else:
            self.output_size = output_size * 2

        super(GrowingDiscriminator, self).__init__()

        self.from_rgb1 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False) #input is 1024 x 1024 x input_channels
        self.block1 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.from_rgb2 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False) #input is 512 x 512 x input_channels
        self.block2 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.from_rgb3 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False) #input is 256 x 256 x input_channels
        self.block3 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.from_rgb4 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False)  # input is 128 x 128 x input_channels
        self.block4 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.from_rgb5 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False)  # input is 64 x 64 x input_channels
        self.block5 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.from_rgb6 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False)  # input is 32 x 32 x input_channels
        self.block6 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.from_rgb7 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False)  # input is 16 x 16 x input_channels
        self.block7 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.from_rgb8 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False)  # input is 8 x 8 x input_channels
        self.block8 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.from_rgb9 = nn.Conv2d(input_channels, layer_channel, kernel_size=1, stride=1, padding=0, bias=False)  # input is 4 x 4 x input_channels
        self.block9 = self.block(layer_channel,layer_channel,use_layer_noise, use_tanh, use_groupnorm, longer_network)

        self.block10= nn.Sequential( nn.Conv2d(layer_channel, layer_channel,kernel_size=4, stride=2,padding=1, bias=False),
                                     nn.Tanh() if use_tanh else nn.LeakyReLU(0.1, True))
        self.conv11 = nn.Conv2d(layer_channel, self.output_size, kernel_size=1, stride=1, padding=0, bias=False)

        self.input_size = starting_size

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.is_transitioning = False;

        self.alpha = 0.0
        self.one_minus_alpha = 1.0

    def forward(self, x):
        last = None
        if(self.input_size==1024):
            if (self.is_transitioning == True):
                last = self.downsample(x)
                last = self.from_rgb2(last)
            x = self.from_rgb1(x)
            x = self.block1(x)
            if (self.is_transitioning == True):
                last = last.mul(self.one_minus_alpha)
                x = x.mul(self.alpha)
                x = x.add(last)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
        elif(self.input_size==512):
            if (self.is_transitioning == True):
                last = self.downsample(x)
                last = self.from_rgb3(last)
            x = self.from_rgb2(x)
            x = self.block2(x)
            if (self.is_transitioning == True):
                last = last.mul(self.one_minus_alpha)
                x = x.mul(self.alpha)
                x = x.add(last)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
        elif (self.input_size == 256):
            if (self.is_transitioning == True):
                last = self.downsample(x)
                last = self.from_rgb4(last)
            x = self.from_rgb3(x)
            x = self.block3(x)
            if (self.is_transitioning == True):
                last = last.mul(self.one_minus_alpha)
                x = x.mul(self.alpha)
                x = x.add(last)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
        elif (self.input_size == 128):
            if (self.is_transitioning == True):
                last = self.downsample(x)
                last = self.from_rgb5(last)
            x = self.from_rgb4(x)
            x = self.block4(x)
            if (self.is_transitioning == True):
                last = last.mul(self.one_minus_alpha)
                x = x.mul(self.alpha)
                x = x.add(last)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
        elif (self.input_size == 64):
            if (self.is_transitioning == True):
                last = self.downsample(x)
                last = self.from_rgb6(last)
            x = self.from_rgb5(x)
            x = self.block5(x)
            if (self.is_transitioning == True):
                last = last.mul(self.one_minus_alpha)
                x = x.mul(self.alpha)
                x = x.add(last)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
        elif (self.input_size == 32):
            if (self.is_transitioning == True):
                last = self.downsample(x)
                last = self.from_rgb7(last)
            x = self.from_rgb6(x)
            x = self.block6(x)
            if (self.is_transitioning == True):
                last = last.mul(self.one_minus_alpha)
                x = x.mul(self.alpha)
                x = x.add(last)
            x = self.block7(x)
            x = self.block8(x)
        elif (self.input_size == 16):
            if (self.is_transitioning == True):
                last = self.downsample(x)
                last = self.from_rgb8(last)
            x = self.from_rgb7(x)
            x = self.block7(x)
            if (self.is_transitioning == True):
                last = last.mul(self.one_minus_alpha)
                x = x.mul(self.alpha)
                x = x.add(last)
            x = self.block8(x)
        elif (self.input_size == 8):
            if(self.is_transitioning==True):
                last = self.downsample(x)
                last = self.from_rgb9(last)
            x = self.from_rgb8(x)
            x = self.block8(x)
            if (self.is_transitioning == True):
                last = last.mul(self.one_minus_alpha)
                x = x.mul(self.alpha)
                x = x.add(last)
        elif (self.input_size == 4):
            x = self.from_rgb9(x)

        x = self.block9(x)
        x = self.block10(x)
        x = self.conv11(x)

        return x.view(x.shape[0],x.shape[1])

    def increase_size(self):
        self.input_size = self.input_size * 2
        if(self.input_size>1024):
            self.input_size = 1024

    def decrease_size(self):
        self.input_size = self.input_size // 2
        if(self.input_size<4):
            self.input_size = 4

    def set_alpha(self, value):
        if value>1.0:
            value = 1.0
        self.alpha = value
        self.one_minus_alpha = 1.0 - value

    def set_transition_time(self, value):
        self.is_transitioning = value
        for param in self.parameters():
            param.requires_grad = True
        if(value == True):
            if (self.input_size >= 1024):
                for param in self.from_rgb2.parameters():
                    param.requires_grad = False
                for param in self.block2.parameters():
                    param.requires_grad = False
            if (self.input_size >= 512):
                for param in self.from_rgb3.parameters():
                    param.requires_grad = False
                for param in self.block3.parameters():
                    param.requires_grad = False
            if (self.input_size >= 256):
                for param in self.from_rgb4.parameters():
                    param.requires_grad = False
                for param in self.block4.parameters():
                    param.requires_grad = False
            if (self.input_size >= 128):
                for param in self.from_rgb5.parameters():
                    param.requires_grad = False
                for param in self.block5.parameters():
                    param.requires_grad = False
            if (self.input_size >= 64):
                for param in self.from_rgb6.parameters():
                    param.requires_grad = False
                for param in self.block6.parameters():
                    param.requires_grad = False
            if (self.input_size >= 32):
                for param in self.from_rgb7.parameters():
                    param.requires_grad = False
                for param in self.block7.parameters():
                    param.requires_grad = False
            if (self.input_size >= 16):
                for param in self.from_rgb8.parameters():
                    param.requires_grad = False
                for param in self.block8.parameters():
                    param.requires_grad = False
            if (self.input_size >= 8):
                for param in self.from_rgb9.parameters():
                    param.requires_grad = False
                for param in self.block9.parameters():
                    param.requires_grad = False
                for param in self.block10.parameters():
                    param.requires_grad = False
                for param in self.conv11.parameters():
                    param.requires_grad = False