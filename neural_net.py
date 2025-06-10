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

class LearningDataset(Dataset):
    def __init__(self, images):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        
        # Определяем трансформации для аугментации
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
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
            nn.Linear(64 * 25 * 25, latent_dim),
        ])
    
    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Decoder, self).__init__()

        self.all_layers = nn.ModuleList([
            nn.Linear(latent_dim, 64 * 25 * 25),
            nn.Unflatten(1, (64, 25, 25)),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 25x25 => 50x50
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 50x50 => 100x100
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        ])
    
    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x

class FullEncoder(nn.Module):
    def __init__(self):
        super(FullEncoder, self).__init__()

        self.all_layers = nn.ModuleList([
            Encoder(1024),
            Decoder(1024)
        ])
    
    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x

    def get_latent(self, x):
        x = self.all_layers[0](x)
        return x

import download_dataset as dd

images = dd.get_images()

full_encoder = FullEncoder().to(device)

dataset = LearningDataset(images)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
optimizer = optim.Adam(full_encoder.parameters(), lr=0.00002)
from torch.optim.lr_scheduler import StepLR
# scheduler = StepLR(optimizer, step_size=5, gamma=0.2)

def get_latent(image):
    with torch.no_grad():
        # Преобразуем изображение в тензор и меняем размерность на (batch, channels, height, width)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # (100, 100, 3) → (1, 3, 100, 100)
        print("Image shape:", image.shape)  # Expecting (1, 3, 100, 100)

        # Пропускаем через encoder
        generated = full_encoder.get_latent(image)
        print("Latent shape:", generated.shape)  # Expecting (1, 64*25*25)
        print("Data min/max:", generated.min().item(), generated.max().item())

    return generated

from PIL import Image

def array_to_image(array):
    img = Image.fromarray(array)
    # Сохраняем изображение в байты
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def combine_images(img1_path_or_array, img2_path_or_array, output_path=None):
    # Если переданы numpy-массивы, преобразуем их в изображения
    if isinstance(img1_path_or_array, np.ndarray):
        img1 = Image.fromarray(((img1_path_or_array + 1) * 127.5).astype(np.uint8)).convert('RGB')  # Предполагаем [-1, 1]
    else:
        img1 = Image.open(img1_path_or_array).convert('RGB')
    
    if isinstance(img2_path_or_array, np.ndarray):
        img2 = Image.fromarray(((img2_path_or_array + 1) * 127.5).astype(np.uint8)).convert('RGB')  # Предполагаем [-1, 1]
    else:
        img2 = Image.open(img2_path_or_array).convert('RGB')

    # Получаем размеры
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Убеждаемся, что высоты совпадают (если нет, обрезаем или растягиваем)
    if height1 != height2:
        new_height = min(height1, height2)
        img1 = img1.crop((0, 0, width1, new_height))
        img2 = img2.crop((0, 0, width2, new_height))

    # Создаём новое изображение
    new_width = width1 + width2
    new_height = height1  # Или height2, если они теперь равны
    new_img = Image.new('RGB', (new_width, new_height))

    # Склеиваем изображения
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (width1, 0))

    # Сохраняем или возвращаем
    if output_path:
        new_img.save(output_path)
    else:
        return new_img

def combine_image_pairs(image_pairs):
    if not image_pairs or not isinstance(image_pairs, list) or not all(len(pair) == 2 for pair in image_pairs):
        raise ValueError("Input must be a list of pairs of numpy arrays [(img1, img2), ...]")

    # Проверяем размеры первого изображения в каждой паре
    height = None
    width = 0
    for i, (img1, img2) in enumerate(image_pairs):
        if not (isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray)):
            raise ValueError(f"Pair {i} contains non-numpy array")
        if len(img1.shape) != 3 or len(img2.shape) != 3 or img1.shape[2] != 3 or img2.shape[2] != 3:
            raise ValueError(f"Pair {i} must have shape (H, W, 3), got {img1.shape} and {img2.shape}")
        if height is None:
            height = max(img1.shape[0], img2.shape[0])
        width += img1.shape[1] + img2.shape[1]

    # Создаём выходной массив
    combined_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Заполняем массив парами
    current_width = 0
    for img1, img2 in image_pairs:
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        # Копируем изображения, обрезая или заполняя, если размеры разные
        h = min(height, h1)
        combined_img[:h, current_width:current_width + w1, :] = img1[:h, :w1, :]
        current_width += w1
        h = min(height, h2)
        combined_img[:h, current_width:current_width + w2, :] = img2[:h, :w2, :]
        current_width += w2

    return combined_img

def one_batch(model, dataloader, criterion, optimizer):
    model.train()
    batch_loss = 0
    batch = next(iter(dataloader))
    batch = batch.to(device)
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, batch)
    loss.backward()
    optimizer.step()
    batch_loss += loss.item()
    print(f'Batch Loss: {loss.item():.4f}')
    
    print(f'Batch completed, Average Loss: {batch_loss:.4f}')
    return batch_loss

def one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        if i == 0 and epoch == 0:
            print(f"Loss at start: {loss.item()}")
        print(f'Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = epoch_loss / len(dataloader)
    print(f'Epoch {epoch+1} completed, Average Loss: {epoch_loss:.4f}')
    # scheduler.step()
    return epoch_loss

latent = get_latent(images[0])
print(latent)
print(latent.shape)

# Image.open(io.BytesIO(array_to_image(images[0]))).show()
# start_image = torch.tensor(images[0], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
def code_and_decode(img_array):
    start_image = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    test_image = full_encoder(start_image).detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
    return test_image
    Image.open(io.BytesIO(array_to_image(test_image))).show()

def tanh_to_img(array):
    return ((array + 1) * 127.5).astype(np.uint8)

# for i in range(1):
#     one_epoch(full_encoder, dataloader, nn.MSELoss(), optimizer, i)
print('images[0].shape')
print(images[0].shape)
print('code_and_decode(images[0]).shape')
print(code_and_decode(images[0]).shape)
print('-------------------')

# Image.open(io.BytesIO(combine_images(images[0], code_and_decode(images[0])))).show()
# Image.open(io.BytesIO(combine_images(images[200], code_and_decode(images[200])))).show()
# Image.open(io.BytesIO(combine_images(images[400], code_and_decode(images[400])))).show()