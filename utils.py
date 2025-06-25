import os
import random
import torch
import torch.nn as nn
from torchvision.utils import save_image

# weight initialization
def weight_init(x):
    classname = x.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(x.weight.data, 0.0, 0.02)
    elif classname.find('InstanceNorm2d') != -1:
        if x.weight is not None:
            nn.init.normal_(x.weight.data, 1.0, 0.02)
        if x.bias is not None:
            nn.init.constant_(x.bias.data, 0.0)

# add replay buffer, essentially saves and feeds older generated images into the discriminator so the generator is't just learning from the most recently generated images
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Buffer size must be > 0" # will give this error message if buffer size is less than zero
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        result = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    result.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    result.append(element)
        return torch.cat(result)
    
def sample_images(batches_done, G_AB, G_BA, val_dataloader, device, save_dir='images'):
    os.makedirs(save_dir, exist_ok=True)
    imgs = next(iter(val_dataloader))
    real_A = imgs['A'].to(device)
    real_B = imgs['B'].to(device)
    
    fake_A = G_BA(real_B)
    fake_B = G_AB(real_A)
    recov_A = G_BA(fake_B)
    recov_B = G_AB(fake_A)

    # combine into a grid for visual comparison
    img_sample = torch.cat((real_A, fake_B, recov_A, real_B, fake_A, recov_B), 0)
    save_image(img_sample, f"{save_dir}/{batches_done}.png", nrow=3, normalize=True)
