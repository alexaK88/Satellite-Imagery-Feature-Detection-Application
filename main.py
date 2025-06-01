import torch
from torch.utils.data import DataLoader
from src.dataset import SatelliteDataset
from src.unet_model import unet_model
from src.train import train_fn
import segmentation_models_pytorch as smp
from albumentations import Resize, Normalize, Compose
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 256

train_transforms = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(),
    ToTensorV2(),
])

val_transforms = train_transforms

train_ds = SatelliteDataset("data/train/images", "data/train/masks", transform=train_transforms)
val_ds = SatelliteDataset("data/val/images", "data/val/masks", transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

model = get_unet().to(DEVICE)
loss_fn = smp.utils.losses.DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
