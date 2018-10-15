import torch
import torch.nn as nn

class NoiseLayer(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            noise = torch.tensor(0).float()
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                noise = noise.to(device)
            if self.is_relative_detach:
                scale = self.sigma * x.detach()
            else:
                scale = self.sigma * x
            sampled_noise = noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

class GrowingGenerator(nn.Module):

    def block(self, input_channels, output_channels, use_noise, use_deconv, use_tanh, use_groupnorm, longer):
        return nn.Sequential(
            NoiseLayer() if use_noise else nn.Sequential(),
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1,
                               bias=False) if use_deconv else nn.Sequential(nn.Upsample(size=None, scale_factor=2, mode='nearest'),
                                                                            nn.Conv2d(input_channels, output_channels,
                                                                                      kernel_size=3, stride=1,
                                                                                      padding=1, bias=False)),
            nn.GroupNorm(1, output_channels) if use_groupnorm else nn.BatchNorm2d(output_channels),
            nn.Tanh() if use_tanh else nn.LeakyReLU(0.1, True),
            nn.Sequential(NoiseLayer() if use_noise else nn.Sequential(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, output_channels) if use_groupnorm else nn.BatchNorm2d(output_channels),
            nn.Tanh() if use_tanh else nn.LeakyReLU(0.1, True)) if longer else nn.Sequential()
        )

    def __init__(self, device,
                 input_class_size,
                 output_channels,
                 starting_size=4,
                 use_tanh_instead_of_relu = True,
                 use_layer_noise = True,
                 use_deconv_instead_of_upsample=True,
                 longer = False,
                 use_groupnorm = True,
                 layer_channel = 256,
                 sampling_size = 128):
        super(GrowingGenerator, self).__init__()
        self.input_class_size = input_class_size
        self.condition = True
        self.layer_channels = layer_channel
        # basic idea:
        # pass input condition through a linear layer
        # combine the two output and pass them through a second linear layer
        if input_class_size <= 1:  # if there is no condition
            self.condition = False
            self.fc_1 = None
            self.fc_2 = nn.Linear(sampling_size, layer_channel, bias=False)
        else:
            self.fc_1 = nn.Linear(self.input_class_size, 128, bias=False)
            self.fc_2 = nn.Linear(128 + sampling_size, layer_channel, bias=False)

        # go through the deconvs
        self.block1 = nn.Sequential(nn.ConvTranspose2d(layer_channel, layer_channel, kernel_size=4, stride=1, padding=0,
                               bias=False), nn.GroupNorm(1, layer_channel), nn.Tanh() if use_tanh_instead_of_relu
                                else nn.LeakyReLU(0.1, True))
        self.to_rgb1 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 4 x 4 x output_channels

        self.block2 = self.block(layer_channel, layer_channel, use_layer_noise, use_deconv_instead_of_upsample, use_tanh_instead_of_relu, use_groupnorm, longer) # out 8 x 8 x layer_channel
        self.to_rgb2 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 8 x 8 x output_channels

        self.block3 = self.block(layer_channel, layer_channel, use_layer_noise, use_deconv_instead_of_upsample, use_tanh_instead_of_relu, use_groupnorm, longer) # out 16 x 16 x layer_channel
        self.to_rgb3 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 16 x 16 x output_channels

        self.block4 = self.block(layer_channel, layer_channel, use_layer_noise, use_deconv_instead_of_upsample, use_tanh_instead_of_relu, use_groupnorm, longer) # out 32 x 32 x layer_channel
        self.to_rgb4 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 32 x 32 x output_channels

        self.block5 = self.block(layer_channel, layer_channel, use_layer_noise, use_deconv_instead_of_upsample, use_tanh_instead_of_relu, use_groupnorm, longer) # out 64 x 64 x layer_channel
        self.to_rgb5 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 64 x 64 x output_channels

        self.block6 = self.block(layer_channel, layer_channel, use_layer_noise, use_deconv_instead_of_upsample, use_tanh_instead_of_relu, use_groupnorm, longer) # out 128 x 128 x layer_channel
        self.to_rgb6 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 128 x 128 x output_channels

        self.block7 = self.block(layer_channel, layer_channel, use_layer_noise, use_deconv_instead_of_upsample, use_tanh_instead_of_relu, use_groupnorm, longer) # out 256 x 256 x layer_channel
        self.to_rgb7 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 256 x 256 x output_channels

        self.block8 = self.block(layer_channel, layer_channel, use_layer_noise, use_deconv_instead_of_upsample, use_tanh_instead_of_relu, use_groupnorm, longer) # out 512 x 512 x layer_channel
        self.to_rgb8 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 512 x 512 x output_channels

        self.block9 = self.block(layer_channel, layer_channel, use_layer_noise, use_deconv_instead_of_upsample, use_tanh_instead_of_relu, use_groupnorm, longer) # out 1024 x 1024 x layer_channel
        self.to_rgb9 = nn.Conv2d(layer_channel, output_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 1024 x 1024 x output_channels

        self.output_activation = nn.Tanh()

        if use_tanh_instead_of_relu:
            self.layer_activation = nn.Tanh()
        else:
            self.layer_activation = nn.LeakyReLU(0.1,inplace=True)

        self.output_size = starting_size

        self.is_transitioning = False

        self.alpha = 0.0
        self.one_minus_alpha = 1.0

        self.upsample = nn.Upsample(size=None, scale_factor=2, mode='nearest')#, align_corners=True)

    def forward(self, x):
        if self.condition == True:
            x1 = x[:, 0:self.input_class_size]
            x2 = x[:, self.input_class_size:]
            x1 = self.layer_activation(self.fc_1(x1))
            x = torch.cat((x1, x2), 1)
        x = self.layer_activation(self.fc_2(x))
        x = x.view(x.shape[0], self.layer_channels, 1, 1)

        last = None
        x = self.block1(x)

        if (self.output_size == 4):
            x = self.to_rgb1(x)
        elif (self.output_size == 8):
            if (self.is_transitioning == True):
                last = self.to_rgb1(x)
            x = self.block2(x)
            x = self.to_rgb2(x)
        elif (self.output_size == 16):
            x = self.block2(x)
            if (self.is_transitioning == True):
                last = self.to_rgb2(x)
            x = self.block3(x)
            x = self.to_rgb3(x)
        elif (self.output_size == 32):
            x = self.block2(x)
            x = self.block3(x)
            if (self.is_transitioning == True):
                last = self.to_rgb3(x)
            x = self.block4(x)
            x = self.to_rgb4(x)
        elif (self.output_size == 64):
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            if (self.is_transitioning == True):
                last = self.to_rgb4(x)
            x = self.block5(x)
            x = self.to_rgb5(x)
        elif (self.output_size == 128):
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            if (self.is_transitioning == True):
                last = self.to_rgb5(x)
            x = self.block6(x)
            x = self.to_rgb6(x)
        elif (self.output_size == 256):
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            if (self.is_transitioning == True):
                last = self.to_rgb6(x)
            x = self.block7(x)
            x = self.to_rgb7(x)
        elif (self.output_size == 512):
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            if (self.is_transitioning == True):
                last = self.to_rgb7(x)
            x = self.block8(x)
            x = self.to_rgb8(x)
        elif (self.output_size == 1024):
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            if (self.is_transitioning == True):
                last = self.to_rgb8(x)
            x = self.block9(x)
            x = self.to_rgb9(x)

        if (self.is_transitioning == True):
            last = self.upsample(last)
            last = last.mul(self.one_minus_alpha)
            x = x.mul(self.alpha)
            x = x.add(last)

        x = self.output_activation(x)
        return x

    def increase_size(self):
        self.output_size = self.output_size * 2
        if (self.output_size > 1024):
            self.output_size = 1024

    def decrease_size(self):
        self.output_size = self.output_size // 2
        if (self.output_size < 4):
            self.output_size = 4

    def set_transition_time(self, value):
        self.is_transitioning = value
        for param in self.parameters():
            param.requires_grad = True
        if (value == True):
            if (self.output_size >= 8):
                if self.condition == True:
                    for param in self.fc_1.parameters():
                        param.requires_grad = False
                for param in self.fc_2.parameters():
                    param.requires_grad = False
                for param in self.block1.parameters():
                    param.requires_grad = False
                for param in self.to_rgb1.parameters():
                    param.requires_grad = False
            if (self.output_size >= 16):
                for param in self.block2.parameters():
                    param.requires_grad = False
                for param in self.to_rgb2.parameters():
                    param.requires_grad = False
            if (self.output_size >= 32):
                for param in self.block3.parameters():
                    param.requires_grad = False
                for param in self.to_rgb3.parameters():
                    param.requires_grad = False
            if (self.output_size >= 64):
                for param in self.block4.parameters():
                    param.requires_grad = False
                for param in self.to_rgb4.parameters():
                    param.requires_grad = False
            if (self.output_size >= 128):
                for param in self.block5.parameters():
                    param.requires_grad = False
                for param in self.to_rgb5.parameters():
                    param.requires_grad = False
            if (self.output_size >= 256):
                for param in self.block6.parameters():
                    param.requires_grad = False
                for param in self.to_rgb6.parameters():
                    param.requires_grad = False
            if (self.output_size >= 512):
                for param in self.block7.parameters():
                    param.requires_grad = False
                for param in self.to_rgb7.parameters():
                    param.requires_grad = False
            if (self.output_size >= 1024):
                for param in self.block8.parameters():
                    param.requires_grad = False
                for param in self.to_rgb8.parameters():
                    param.requires_grad = False

    def set_alpha(self, value):
        if value > 1.0:
            value = 1.0
        self.alpha = value
        self.one_minus_alpha = 1.0 - value