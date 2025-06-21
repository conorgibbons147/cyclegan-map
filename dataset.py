import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        super().__init__()
        self.transform = transform
        self.mode = mode

        self.dir_A = os.path.join(root, 'modern') # path for modern images
        self.dir_B = os.path.join(root, 'vintage') # same for vintage

        self.files_A = sorted(os.listdir(self.dir_A)) # lists image filenames in each of the folders in order
        self.files_B = sorted(os.listdir(self.dir_B))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    def __getitem__(self, index):
        img_A = Image.open(os.path.join(self.dir_A, self.files_A[index % len(self.files_A)])).convert('RGB')
        img_B = Image.open(os.path.join(self.dir_B, self.files_B[index % len(self.files_B)])).convert('RGB')

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        
        return {'A': img_A, 'B': img_B}
    
    