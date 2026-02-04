import torch
from sit_sp import SmallREG
from training_sp import Diffusion, train

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = "./data/cifar10" 
    
    #initializae model
    model = SmallREG(
        input_size=32,
        patch_size=2,
        hidden_size=384, # 원래 크기
        depth=28,        # 원래 크기
        num_heads=12,
        num_classes=10
    ).to(device)
    
    # Diffusion
    diffusion = Diffusion(T=1000, num_classes=10)
    
    # train
    train(
        data_path=data_path,
        model=model,
        diffusion=diffusion,
        epochs=100,      # 충분한 학습 횟수
        batch_size=128,  # GPU 메모리에 맞춰 조절 (보통 64~256)
        lr=3e-4
    )

if __name__ == "__main__":
    main()