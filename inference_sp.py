import torch
from sit_sp import SmallREG
from training_sp import Diffusion
from torchvision.utils import save_image, make_grid
import os
import numpy as np

def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SmallREG(input_size=32,patch_size=2, in_channels=3,hidden_size=384, depth=16).to(device)
    model_path = args.model_path
    batch_size = args.batch_size
    num_samples = args.num_samples


    print(f"loading model from {model_path}")
    if not os.path.exists(model_path):    
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    checkpoint = torch.load(model_path,map_location=device)

    if 'ema_state_dict' in checkpoint:
        print("✅ EMA weights found! Loading EMA state_dict for better generation quality...")
        model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
    elif 'model_state_dict' in checkpoint:
        print("⚠️ EMA weights NOT found. Loading standard model_state_dict...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    diffusion = Diffusion(num_classes=10)
    print(f" sampling started .... target : {num_samples} images")
    all_samples = []
    time_stats = []
    num_batches = num_samples// batch_size

    for i in range(num_batches):
        labels = torch.arange(0,10).to(device)
        labels = labels.unsqueeze(0).repeat(batch_size//10,1)

        samples, duration = diffusion.sample(model, batch_size,labels, cfg_scale=4.0)
        all_samples.append(samples.cpu())
        time_stats.append(duration)

        print(f"   Batch {i+1}/{num_batches} completed in {duration:.2f} sec")
    

    avg_time = np.mean(time_stats)
    min_time = np.min(time_stats)
    max_time = np.max(time_stats)
    total_duration = np.sum(time_stats)

    print("-" * 50)
    print(f"📊 Performance Statistics ({num_samples} images, {batch_size} per batch)")
    print(f"   Total Time : {total_duration:.2f} sec")
    print(f"   Avg Time per Batch ({batch_size} imgs): {avg_time:.4f} sec")
    print(f"   Min Time per Batch : {min_time:.4f} sec")
    print(f"   Max Time per Batch : {max_time:.4f} sec")
    print(f"   Avg Time per Image : {avg_time / batch_size:.4f} sec")
    print("-" * 50)



    os.makedirs("results", exist_ok=True)
    final_images = torch.cat(all_samples,dim=0)
    grid = make_grid(final_images, nrow=10, normalize=False)
    save_path = "results/generated_stats_sample.png"
    save_image(grid, save_path)

    # print(f"💾 Saved all {num_samples} generated images to {save_path}")
    # save_image(samples, "results/generated_sample.png", nrow=4,normalize=False)
    # print("Saved generated images to results/generated_sample.png")
