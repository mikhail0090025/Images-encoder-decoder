import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DogsDataset(Dataset):
    def __init__(self, images):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        
        # Определяем трансформации для аугментации
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(1),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # if self.transform:
        #     image = self.transform(image)
        return image

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.all_layers = nn.ModuleList([
            # 100x100 => 50x50
            nn.Conv2d(3, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, 3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 50x50 => 25x25
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Tanh(),
        ])
    
    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.all_layers = nn.ModuleList([
            # Input
            nn.Unflatten(1, (64, 25, 25)),
            
            # 25x25 => 50x50
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 50x50 => 100x100
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Last layer
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Conv2d(3, 3, 1),
            nn.Tanh(),
        ])
    
    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x

import download_dataset as dd

images = dd.get_images()

encoder = Encoder().to(device)
decoder = Decoder().to(device)

def get_latent(image):
    with torch.no_grad():
        # Преобразуем изображение в тензор и меняем размерность на (batch, channels, height, width)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # (100, 100, 3) → (1, 3, 100, 100)
        print("Image shape:", image.shape)  # Expecting (1, 3, 100, 100)

        # Пропускаем через encoder
        generated = encoder(image)
        print("Latent shape:", generated.shape)  # Expecting (1, 64*25*25)
        print("Data min/max:", generated.min().item(), generated.max().item())

        # Пропускаем через decoder для восстановления
        reconstructed = decoder(generated)
        print("Reconstructed shape:", reconstructed.shape)  # Expecting (1, 3, 100, 100)

        # Преобразуем обратно в numpy для отображения
        reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (1, 3, 100, 100) → (100, 100, 3)

    return generated, reconstructed

latent, reconstructered = get_latent(images[0])
print(latent)
print(latent.shape)
print(reconstructered.shape)

def one_epoch(batch_size):
    batches_count = images.shape[0] / batch_size