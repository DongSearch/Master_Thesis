import torch.optim as optim
import dataset_sp as dt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from sit_sp import SmallREG
from torch.utils.data import DataLoader, TensorDataset
import wandb
import time
import os
import argparse
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler, autocast

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: torchmetrics not installed. FID will be skipped.")



device = "cuda" if torch.cuda.is_available() else "cpu"

class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02,num_classes=10):
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T).to(device) # noise get harder as step goes
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha,dim=0) # cumulative production of alpha
        self.num_classes= num_classes

    def noise_images(self,x,t):
        """
        Docstring for noise_images
        
        :param x: (B,C,H,W)
        :param t: (B,)
        return : x_t(Noisy image), noise(Target noise)
        """
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None,None,None]
        sqrt_one_mius_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]

        epsilon = torch.randn_like(x)
        x_t = sqrt_alpha_bar * x + sqrt_one_mius_alpha_bar * epsilon
        return x_t, epsilon
    
    def sample(self,model,batch_size,labels,cfg_scale=3.0):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            x = torch.randn((batch_size,3,32,32)).to(device) # random noise
            # null label generation for CFG
            null_labels = torch.full_like(labels,self.num_classes).to(device)
            # T : 999 -> 1
            for i in tqdm(reversed(range(0,self.T)), position=0):
                t = (torch.ones(batch_size)*i).long().to(device)

                #CFG
                combined_x = torch.cat([x,x],dim=0)
                combined_t = torch.cat([t,t],dim=0)
                combined_y = torch.cat([labels, null_labels],dim=0)

                # model prediciton
                model_out = model(combined_x,combined_t,combined_y)
                # seperation consequence
                eps_cond, eps_uncond = torch.chunk(model_out,2,dim=0)
                predicted_noise = eps_uncond + cfg_scale*(eps_cond - eps_uncond)

                alpha = self.alpha[t][:, None,None,None]
                alpha_bar = self.alpha_bar[t][:, None,None,None]
                beta = self.beta[t][:, None,None,None]
                # if it is rastt step, no add noise, otherwise put random noise to avoid getting distorted by substration noise

                if i > 0 :
                    noise = torch.randn_like(x)
                else :
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise
        # time measurement
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Total sampling time for {batch_size} images: {total_time:.2f} sec")
        print(f"Speed: {batch_size / total_time:.2f} images/sec")

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x, total_time

def train(data_path,model, diffusion, epochs= 100, batch_size=64, lr=3e-4,resume_path = None, save_dir='checkpoints'):
    # intialization wandb
    wandb.init(project="REG-CIFAR10", mode = "offline",
               config={
                        "epochs" : epochs,
                        "batch_size" : batch_size,
                        "lr" : lr,
                        "architecture" : "SmallREG"
                    }
            )
    os.makedirs(save_dir, exist_ok=True)


    train_loader = dt.train_pre_processing(data_path,batch_size=batch_size)
    val_loader = dt.test_pre_processing(data_path,batch_size)
    scaler = GradScaler()
    opt = optim.AdamW(model.parameters(),lr=lr)
    criterion = nn.MSELoss()

    fid = FrechetInceptionDistance(feature=64).to(device) if HAS_FID else None


    start_epoch = 0
    # before start, it checks resume_file
    if resume_path is not None and os.path.exists(resume_path):
        print(f"resuming training from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"loaded check point! resuming from epoch{start_epoch + 1}")
    else :
        print("starting training from scratch")

    model.train()
    for epoch in range(start_epoch,epochs):
        pbar = tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}")
        avg_train_loss = 0
        epoch_loss = 0

        for i, (imgs,lbs) in enumerate(pbar):
            imgs = imgs.to(device)
            lbs = lbs.to(device)
            B = imgs.shape[0]

            t = torch.randint(0,diffusion.T,(B,), device=device).long()

            with autocast():
                x_t, noise = diffusion.noise_images(imgs,t)# forward diffusion
                predicted_noise = model(x_t,t,lbs) # prediction
                loss = criterion(predicted_noise,noise)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
            wandb.log({"batch_mse":loss.item()})
        
        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for v_imgs, v_lbs in val_loader :
                v_imgs, v_lbs = v_imgs.to(device), v_lbs.to(device)
                v_t = torch.randint(0, diffusion.T, (v_imgs.shape[0],), device=device)
                with autocast():
                    v_xt, v_noise = diffusion.noise_images(v_imgs,v_t)
                    v_pred = model(v_xt, v_t,v_lbs)
                    val_loss = criterion(v_pred, v_noise)
                val_epoch_loss += val_loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train MSE: {avg_train_loss:.4f}, Val MSE: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_mse": avg_train_loss, "val_mse": avg_val_loss})
        
        
        test_labels = torch.arange(0, 10).to(device)  # 10개 클래스 샘플링
        samples, _ = diffusion.sample(model, 10, test_labels, cfg_scale=0.0) # T=1000 고품질

        # 이미지 그리드 생성 및 저장
        # save_dir 내부에 'samples' 폴더 생성
        sample_img_dir = os.path.join(save_dir, 'samples')
        os.makedirs(sample_img_dir, exist_ok=True)
        
        grid_save_path = os.path.join(sample_img_dir, f"sample_epoch_{epoch+1}.png")
        vutils.save_image(samples, grid_save_path, nrow=5)
        print(f"Sample image saved to {grid_save_path}")
            


        if (epoch + 1) % 10 == 0 :
            # test_labels = torch.arange(0,10).to(device)
            # samples, speed = diffusion.sample(model,10,test_labels)
            # images = [wandb.Image(img) for img in samples]
            # wandb.log({"sampled_images": images, "sampling_speed_sec": speed})

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(), 
                'loss': avg_train_loss,
            }
            save_path = os.path.join(save_dir, f"reg_cifar10_ep{epoch+1}.pth")
            torch.save(checkpoint, save_path)
            print(f"checkpoint saved to {save_path}")
            print("Generating samples for FID and Visualization...")


            
            # 1. (Reset -> Add Real -> Add Fake -> Compute)
            if HAS_FID:
                fid.reset() 
                fid_real_limit_batches = 10 
                with torch.no_grad():
                    for i, (real_imgs, _) in enumerate(val_loader):
                        if i >= fid_real_limit_batches: break
                        real_imgs = real_imgs.to(device)
                        # [-1, 1] -> [0, 1] -> [0, 255] uint8
                        real_imgs = (real_imgs.clamp(-1, 1) + 1) / 2
                        real_imgs = (real_imgs * 255).to(torch.uint8)
                        fid.update(real_imgs, real=True)

                num_fid_samples = 128
                fake_batches = num_fid_samples // batch_size
                with torch.no_grad():
                    for _ in tqdm(range(fake_batches), desc="FID generation"):
                        labels = torch.randint(0, diffusion.num_classes, (batch_size,), device=device)
                        with autocast():
                            fake_imgs, _ = diffusion.sample(model, batch_size, labels)
                        fake_imgs = (fake_imgs * 255).to(torch.uint8)
                        fid.update(fake_imgs, real=False)

                fid_score = fid.compute().item()
                print(f"Epoch {epoch+1} FID: {fid_score:.4f}")
                wandb.log({"fid": fid_score})

    wandb.finish()
