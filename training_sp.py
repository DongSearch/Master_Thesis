import torch.optim as optim
import dataset_sp as dt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from sit_sp import SmallREG
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha,dim=0) # cumulative production of alpha

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
        with torch.no_grad():
            x = torch.randn((batch_size,3,32,32)).to(device) # random noise
            # T : 999 -> 1
            for i in tqdm(reversed(range(1,self.T)), position=0):
                t = (torch.ones(batch_size)*i).long().to(device)

                predicted_noise = model(x,t,labels)
                alpha = self.alpha[t][:, None,None,None]
                alpha_bar = self.alpha_bar[t][:, None,None,None]
                beta = self.beta[t][:, None,None,None]
                # if it is rastt step, no add noise, otherwise put random noise to avoid getting distorted by substration noise

                if i > 1 :
                    noise = torch.randn_like(x)
                else :
                    noise = torch.zeros_like(x)

        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        # [-1, 1] -> [0, 1]로 변환하여 리턴
        x = (x.clamp(-1, 1) + 1) / 2
        return x

def train(data_path,model, diffusion, epochs= 100, batch_size=64, lr=3e-4):
    train_loader, val_loader = dt.train_val_pre_processing(data_path,batch_size=batch_size)
    opt = optim.AdamW(model.parameters(),lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}")
        avg_loss = 0

        for i, (imgs,lbs) in enumerate(pbar):
            imgs = imgs.to(device)
            lbs = lbs.to(device)
            B = imgs.shape[0]

            t = torch.randint(0,diffusion.T,(B,), device=device).long()
            x_t, noise = diffusion.noise_images(imgs,t)# forward diffusion
            predicted_noise = model(x_t,t,lbs) # prediction
            loss = criterion(predicted_noise,noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            avg_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {avg_loss / len(train_loader):.4f}")

        if (epoch + 1) % 10 == 0 :
            torch.save(model.state_dict(), f"reg_cifar10_ep{epoch+1}.pth")





# def get_dummy_loader(batch_size=4):
#     # CIFAR10 크기: (Batch, 3, 32, 32)
#     imgs = torch.randn(16, 3, 32, 32) 
#     labels = torch.randint(0, 10, (16,))
#     dataset = TensorDataset(imgs, labels)
#     return DataLoader(dataset, batch_size=batch_size)

# def test_pipeline():
#     # 1. 설정 (CPU 모드)
#     device = "cpu"
#     print(f"Testing on {device}...")
    
#     # 2. 모델 초기화
#     model = SmallREG(
#         input_size=32, patch_size=2, in_channels=3, 
#         hidden_size=64, depth=4, num_heads=4, num_classes=10 # 테스트용으로 작게 축소
#     ).to(device)
  
#     diffusion = Diffusion(T=10) # T를 작게 설정
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#     criterion = nn.MSELoss()

#     # 3. 데이터 로드
#     train_loader = get_dummy_loader(batch_size=4)

#     # 4. 학습 루프 (1 Epoch만)
#     model.train()
#     for i, (imgs, lbs) in enumerate(train_loader):
#         imgs, lbs = imgs.to(device), lbs.to(device)
        
#         t = torch.randint(0, diffusion.T, (imgs.shape[0],), device=device).long()
#         x_t, noise = diffusion.noise_images(imgs, t)
        
#         # Forward
#         predicted_noise = model(x_t, t, lbs)
        
#         # Loss Check
#         # 모델 출력이 (B, 6, H, W)라면 여기서 에러 발생함. (B, 3, H, W)여야 함.
#         if predicted_noise.shape != noise.shape:
#             print(f"❌ Shape Mismatch! Pred: {predicted_noise.shape}, Target: {noise.shape}")
#             print("Tip: sit_sp.py의 self.out_channels = in_channels * 2 를 * 1로 수정하세요.")
#             break
            
#         loss = criterion(predicted_noise, noise)
        
#         # Backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         print(f"Batch {i}: Loss {loss.item():.4f} (Pass ✅)")

# if __name__ == "__main__":
#     test_pipeline()




def test_sampling_cpu():
    # 1. 모델 준비 (작게 축소)
    model = SmallREG(
        input_size=32, patch_size=2, in_channels=3, 
        hidden_size=64, depth=2, num_heads=4, num_classes=10
    ).to(device)
    
    # ★ 중요: sit_sp.py를 수정하지 않았다면 여기서 강제 수정 (Shape 맞추기용)
  # 2. Diffusion 준비 (빠른 테스트를 위해 T=10으로 설정)
  
    diffusion = Diffusion(T=10)

    # 3. 더미 데이터
    batch_size = 2
    dummy_labels = torch.randint(0, 10, (batch_size,)).to(device)

    # 4. 샘플링 실행
    try:
        generated_images = diffusion.sample(model, batch_size, dummy_labels)
        print("\n✅ Sampling Success!")
        print(f"Output Shape: {generated_images.shape}") # (2, 3, 32, 32)
        print(f"Output Range: [{generated_images.min():.4f}, {generated_images.max():.4f}] (Should be 0~1)")
    except Exception as e:
        print(f"\n❌ Sampling Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sampling_cpu()