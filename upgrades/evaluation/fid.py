import os
import cv2
import torch
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import linalg
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from inception import InceptionV3

# Define frame extraction function

def extract_frames(video_path, frame_folder, target_frame_count=None):
    """Extract frames from a video file and save them to a folder."""
    os.makedirs(frame_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If a target frame count is specified, calculate step to downsample
    if target_frame_count:
        step = total_frames // target_frame_count
    else:
        step = 1
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame_filename = os.path.join(frame_folder, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_idx += 1
    
    cap.release()
    
    return total_frames

# Image dataset class
class ImagePathDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        img = cv2.imread(str(self.files[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return self.transform(img) if self.transform else img

# Get activations from InceptionV3
def get_activations(files, model, batch_size, dims, device):
    """Get activations from InceptionV3 model for FID computation."""
    model.eval()
    dataset = ImagePathDataset(files, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    pred_arr = np.empty((len(files), dims))
    start_idx = 0

    for batch in tqdm(data_loader, desc="Extracting Features"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx += pred.shape[0]

    return pred_arr

# Calculate FID
def calculate_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    """Compute the Frechet Inception Distance (FID)."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @ (sigma2 + eps * np.eye(sigma2.shape[0])))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

# Compute activations for a given path
def compute_statistics_of_path(path, model, batch_size, dims, device):
    """Compute mean and covariance of activations for FID."""
    path = Path(path)
    files = sorted(path.glob("*.jpg"))
    
    if len(files) == 0:
        raise ValueError(f"No images found in {path}")

    act = get_activations(files, model, batch_size, dims, device)
    mu, sigma = np.mean(act, axis=0), np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid_given_paths(path1, path2, batch_size, dims, device='cpu'):
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    mu1, sigma1 = compute_statistics_of_path(path1, model, batch_size, dims, device)
    mu2, sigma2 = compute_statistics_of_path(path2, model, batch_size, dims, device)
    
    fid_value = calculate_frechet_distance(mu1, mu2, sigma1, sigma2)
    return print('FID distance:', round(fid_value, 3))

# End-to-end FID calculation for videos
def calculate_fid_for_videos(real_video, generated_video, batch_size=32, dims=2048, device='cpu', cleanup=True):
    """Extract frames from videos, compute FID, and optionally clean up extracted images."""
    
    src_path, gen_path = "src_frames", "gen_frames"
    
    print("Extracting frames from real video...")
    real_frame_count = extract_frames(real_video, src_path)
    
    print("Extracting frames from generated video...")
    gen_frame_count = extract_frames(generated_video, gen_path)

    # Ensure both videos have the same number of frames
    if real_frame_count != gen_frame_count:
        min_frame_count = min(real_frame_count, gen_frame_count)
        print(f"Warning: Frame counts differ. Trimming to {min_frame_count} frames.")
        
        # Trim both to the minimum frame count
        trim_frames(src_path, min_frame_count)
        trim_frames(gen_path, min_frame_count)
    
    print("\nCalculating FID...")
    fid_score = calculate_fid_given_paths(src_path, gen_path, batch_size, dims, device)

    # Cleanup extracted frames
    if cleanup:
        shutil.rmtree(src_path, ignore_errors=True)
        shutil.rmtree(gen_path, ignore_errors=True)
        print("Temporary frame directories cleaned up.")

    return fid_score

def trim_frames(frame_folder, target_frame_count):
    """Trim frames in a folder to the target frame count."""
    frames = sorted(os.listdir(frame_folder))
    for i in range(target_frame_count, len(frames)):
        os.remove(os.path.join(frame_folder, frames[i]))

# Run the script
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    real_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/data/raw/videos/May.mp4"
    generated_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/generated-videos/may_infer.mp4"

    print("\nStarting FID computation...")
    calculate_fid_for_videos(real_video_path, generated_video_path, device=device)
