import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False):
        self.root = root  # Store the root directory
        self.unaligned = unaligned

        # List all image files in each dataset folder
        self.files_A = sorted(os.listdir(os.path.join(root, 'train_A')))  # Non-shadow region images
        self.files_B = sorted(os.listdir(os.path.join(root, 'train_B')))  # Shadow region images
        self.files_C = sorted(os.listdir(os.path.join(root, 'train_C')))  # Mask images
        self.files_D = sorted(os.listdir(os.path.join(root, 'train_D')))  # Non-shadow
        self.files_E = sorted(os.listdir(os.path.join(root, 'train_E')))  # Real shadow region + non-shadow
        self.files_F = sorted(os.listdir(os.path.join(root, 'train_F')))  # Dilated mask

        # Define the transform pipeline for the images
        self.transform = transforms.Compose([
            transforms.Resize((400, 400), Image.BICUBIC),  # Resize to 400x400
            transforms.RandomHorizontalFlip(),  # Data augmentation (random horizontal flip)
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Data augmentation (color)
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image (mean and std dev)
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((400, 400), Image.BICUBIC),  # Resize mask
            transforms.RandomHorizontalFlip(),  # Random flip for mask as well
            transforms.ToTensor()  # Convert mask to tensor
        ])

    def __getitem__(self, index):
        # Open images from respective folders
        item_A = self.transform(Image.open(os.path.join(self.root, 'A', self.files_A[index % len(self.files_A)])))
        item_B = self.transform(Image.open(os.path.join(self.root, 'B', self.files_B[random.randint(0, len(self.files_B) - 1)])))
        item_C = self.transform_mask(Image.open(os.path.join(self.root, 'C', self.files_C[index % len(self.files_C)])).convert('L'))  # Grayscale mask
        item_D = self.transform(Image.open(os.path.join(self.root, 'D', self.files_D[index % len(self.files_D)])))
        item_E = self.transform(Image.open(os.path.join(self.root, 'E', self.files_E[index % len(self.files_E)])))
        item_F = self.transform_mask(Image.open(os.path.join(self.root, 'F', self.files_F[index % len(self.files_F)])))

        return {'A': item_A, 'B': item_B, 'C': item_C, 'D': item_D, 'E': item_E, 'F': item_F}

    def __len__(self):
        # Return the size of the largest dataset to ensure full use of all data
        return max(len(self.files_A), len(self.files_B), len(self.files_C), len(self.files_D), len(self.files_E), len(self.files_F))
