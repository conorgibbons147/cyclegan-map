import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from models import Generator  # <-- Import your generator class

# load the trained generator model
CHECKPOINT_PATH = "G_AB_trained.pth"

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
    title="Map → Vintage Converter",
    description="Upload a modern map image to get a vintage-style version using CycleGAN."
)

if __name__ == "__main__":
    iface.launch()
