import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residuals=9):
        super().__init__()

        model = [] # list of the layers we add in order
        
        # create an initial convolution block
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7), # 7x7 convolution filter for context, goes from 3 to 64 channels (feature maps)
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # Now we downsample layers twice. This reduces H and W and increases channels
        in_features = 64
        out_features = 128
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(num_residuals):                  # keeps feature map structure/size
            model += [ResidualBlock(in_features)]
        
        # upsampling twice to reverse downsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

            # final output layer
            model += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, out_channels, 7)
                nn.Tanh()
            ]
        self.model = nn.Sequential(*model) # the * before model unpacks the list of layers so that pytorch can see each one

    def forward(self, x):
        return self.model(x)     