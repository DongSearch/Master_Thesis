import torch
from sit_sp import SmallREG
from training_sp import Diffusion
from torchvision.utils import save_image, make_grid
import os
import numpy as np

def run_inference(model_path, num_samples=16, batch_size=10, device="cuda"):
    model = SmallREG(input_size=32,patch_size=2, in_channels=3,hidden_size=384, depth=28).to(device)

    print(f"loading model from {model_path}")
    if not os.path.exists(model_path):    
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    checkpoint = torch.load(model_path,map_location=device)

    if 'model_state_dict' in checkpoint :
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.loade_state_dict(checkpoint)
    
    model.eval()
    diffusion = Diffusion(num_classes=10)
    print(f" sampling started .... target : {num_samples} images")
    all_samples = []
    time_stats = []
    num_batches = num_samples// batch_size

    for i in range(num_batches):
        labels = torch.randint(0,10,(batch_size,)).to(device)
        samples, duration = diffusion.sample(model, batch_size,labels, cfg_scale=3.0)
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

    print(f"💾 Saved all {num_samples} generated images to {save_path}")
    save_image(samples, "result/generated_sample.png", nrow=4,normalize=False)
    print("Saved generated images to results/generated_sample.png")
