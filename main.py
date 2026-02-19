import argparse
from training_sp import train, Diffusion
from sit_sp import SmallREG
import inference_sp
import torch
import os
def main():
    parser = argparse.ArgumentParser(description="SmallREG Diffusion Model on CIFAR-10")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '../../data/cifar10_images'))
    # 모드 선택 (필수)
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], help='Choose mode: train or inference')
    
    # 공통 설정
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_type', type=str, default='cifar_10', help='data type')
    parser.add_argument('--high_block_proportion', type=float, default=0.5, help='data type')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')

    
    # Training 관련 설정
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to .pth file to resume training')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB for logging')

    # Inference 관련 설정
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model for inference')
    parser.add_argument('--num_samples', type=int, default=100, help='Total number of images to generate')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data_type == "cifar_10":
        C,H,W,N,D,NH = 3,32,32,10,16,12
    elif args.data_type == "mnist":
        C,H,W,N,D,NH = 1,28,28,10,6,6

    if args.mode == 'train':
        print(f"🔥 Starting Training on {device}...")
        model = SmallREG(input_size=H, patch_size=2, in_channels=C, hidden_size=384, depth=D, num_heads = NH,
                         high_low_split=args.high_block_proportion, split_threshold= args.threshold, overlap=0.2).to(device)
        diffusion = Diffusion(num_classes=N)
        train(
        args.data_path, 
        model, 
        diffusion, 
        args.epochs, 
        args.batch_size, 
        args.lr, 
        args.resume_path, 
        args.save_dir,
        C,
        H,
        W,
        N,
    )

    elif args.mode == 'inference':
        if args.model_path is None:
            raise ValueError("❌ Error: --model_path is required for inference mode.")
        
        print(f"🎨 Starting Inference on {device}...")
        inference_sp.run_inference(args)

if __name__ == "__main__":
    main()