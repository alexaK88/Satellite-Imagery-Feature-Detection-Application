import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)
    model.train()
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())
