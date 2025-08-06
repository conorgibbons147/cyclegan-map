import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import sys
import os
import torch
import torch.nn as nn

# for now due to file path issues, I added the entire generator class code so it can be called in the app.py file
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
    def __init__(self, in_channels=3, out_channels=3, num_residuals=6):
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
        out_features = in_features // 2
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
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
            ]
        
        self.model = nn.Sequential(*model) # the * before model unpacks the list of layers so that pytorch can see each one

    def forward(self, x):
        return self.model(x)     
    

# load the trained generator model
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "G_AB_epoch10.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_AB = Generator().to(device)
G_AB.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
G_AB.eval()

# process the input image
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # match training size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# function to map modern map to vintage style
def map_to_vintage(input_image):
    image = preprocess(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        fake_vintage = G_AB(image)
    fake_vintage = (fake_vintage * 0.5) + 0.5
    return transforms.ToPILImage()(fake_vintage.squeeze().cpu())

# gradio interface for the app
iface = gr.Interface(
    fn=map_to_vintage,
    inputs=gr.Image(type="pil", label="Upload Modern Map"),
    outputs=gr.Image(type="pil", label="Vintage Map"),
    title="Map Converter",
    description="Upload a modern map image to get a vintage-style version using CycleGAN."
)

if __name__ == "__main__":
    iface.launch()
