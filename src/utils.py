import torch
import matplotlib.pyplot as plt

def save_predictions(loader, model, device, folder="preds/"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.5).float()

        plt.imsave(f"{folder}/pred_{idx}.png", preds.squeeze().cpu().numpy(), cmap="gray")
    model.train()
