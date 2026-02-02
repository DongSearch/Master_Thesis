import torch.optim as optim
import dataset_sp as dt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(data_path,model, diffusion, epochs= 100, batch_size=64, lr=3e-4):
    train_loader = dt.train_val_pre_processing(data_path,batch_size=batch_size,lr=lr)
    opt = optim.AdamW(model.parameters(),lr=lr)
    loss = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}")
        avg_loss = 0

        for i, (imgs,lbs) in enumerate(pbar):
            imgs = imgs.to(device)
            lbs = lbs.to(device)
            B = imgs.shape[0]

            t = torch.randint(0,diffusion.T,(B,), device=device).long()