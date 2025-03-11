import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import dlib
import torch.nn.functional as F
from transformers import AutoModel  # For loading pre-trained models

# Load videos
real_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/data/raw/videos/May.mp4"
generated_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/generated-videos/May_demo.mp4"

real_vid = cv2.VideoCapture(real_video_path)
gen_vid = cv2.VideoCapture(generated_video_path)

# Get video properties
fps_real = real_vid.get(cv2.CAP_PROP_FPS)
fps_gen = gen_vid.get(cv2.CAP_PROP_FPS)

print(f"Real Video FPS: {fps_real}, Generated Video FPS: {fps_gen}")

# Load LPIPS model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim_scores = []
psnr_scores = []
lpips_scores = []
fid_scores = []
lmd_scores = []
sync_loss_scores = []

# Load dlib's face landmark detector
predictor_path = "shape_predictor_68_face_landmarks.dat"  # You might need to download this
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load SyncNet Model from Hugging Face (Placeholder - Replace with actual model name)
# Example: Replace 'YOUR_SYNCNET_MODEL_NAME' with the actual model name from Hugging Face
try:
    syncnet_model = AutoModel.from_pretrained("joonson/syncnet_mel").to(device)
except Exception as e:
    print(f"Error loading SyncNet model: {e}")
    syncnet_model = None # Set to None if loading fails

if syncnet_model is not None:
    syncnet_model.eval()

# Process frames one by one (avoid storing all frames in memory)
while True:
    ret_real, frame_real = real_vid.read()
    ret_gen, frame_gen = gen_vid.read()
    
    if not ret_real or not ret_gen:
        break
    
    # Convert frames to RGB
    frame_real = cv2.cvtColor(frame_real, cv2.COLOR_BGR2RGB)
    frame_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2RGB)

    # Compute SSIM (convert to grayscale)
    real_gray = cv2.cvtColor(frame_real, cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(frame_gen, cv2.COLOR_RGB2GRAY)
    ssim_score = ssim(real_gray, gen_gray, data_range=gen_gray.max() - gen_gray.min())
    ssim_scores.append(ssim_score)

    # Compute PSNR
    mse = np.mean((frame_real - frame_gen) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100
    psnr_scores.append(psnr)

    # Compute LPIPS

    real_torch = torch.tensor(frame_real, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
    gen_torch = torch.tensor(frame_gen, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
    real_torch = real_torch.to(device)
    gen_torch = gen_torch.to(device)

    with torch.no_grad():
        perceptual_loss = lpips.LPIPS(net='alex').to(device)
        lpips_score = perceptual_loss(real_torch, gen_torch).item()
    lpips_scores.append(lpips_score)

    # Compute FID (Frechet Inception Distance)
    fid_score = calculate_fid(frame_real, frame_gen)
    fid_scores.append(fid_score)

    # Compute LMD (Lip Movement Distance)
    lmd_score = calculate_lmd(frame_real, frame_gen, detector, predictor)
    lmd_scores.append(lmd_score)

    # Compute Sync Loss
    if syncnet_model is not None:
        sync_loss_score = calculate_sync_loss(frame_real, frame_gen, syncnet_model, device)
        sync_loss_scores.append(sync_loss_score)
    else:
        sync_loss_scores.append(0) # Append 0 if model loading failed

real_vid.release()
gen_vid.release()

# Print results
print(f"SSIM: {np.mean(ssim_scores):.4f}")
print(f"PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"LPIPS: {np.mean(lpips_scores):.4f} (Lower is better)")
print(f"FID: {np.mean(fid_scores):.2f}")
print(f"LMD: {np.mean(lmd_scores):.4f}")
print(f"Sync Loss: {np.mean(sync_loss_scores):.4f}")

def calculate_fid(img1, img2):
    # Resize images to 299x299
    img1 = cv2.resize(img1, (299, 299))
    img2 = cv2.resize(img2, (299, 299))

    # Transform to torch tensors
    transform = transforms.ToTensor()
    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)

    # Load pre-trained InceptionV3 model
    inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    inception_model.eval()
    inception_model.to(device)

    # Get activations
    with torch.no_grad():
        act1 = inception_model(img1_tensor.to(device))[0].cpu().numpy().flatten()
        act2 = inception_model(img2_tensor.to(device))[0].cpu().numpy().flatten()

    # Calculate mean and covariance
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_lmd(img1, img2, detector, predictor):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces1 = detector(gray1)
    faces2 = detector(gray2)

    if len(faces1) == 0 or len(faces2) == 0:
        return 0  # Return 0 if no faces are detected in either image

    # Get lip landmarks (landmarks 49-68)
    landmarks1 = predictor(gray1, faces1[0])
    landmarks2 = predictor(gray2, faces2[0])

    # Extract lip landmarks
    lip_landmarks1 = np.array([[p.x, p.y] for p in landmarks1.parts()[48:68]])
    lip_landmarks2 = np.array([[p.x, p.y] for p in landmarks2.parts()[48:68]])

    # Calculate Euclidean distance
    lmd = np.mean(np.sqrt(np.sum((lip_landmarks1 - lip_landmarks2) ** 2, axis=1)))
    return lmd

def calculate_sync_loss(img, frame_gen, syncnet_model, device):
    # 1. Extract Lip Region:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if not faces:
        return 0  # No face detected

    landmarks = predictor(gray, faces[0])
    lip_points = landmarks.parts()[48:68]  # Lip landmark indices
    min_x = min(p.x for p in lip_points)
    max_x = max(p.x for p in lip_points)
    min_y = min(p.y for p in lip_points)
    max_y = max(p.y for p in lip_points)

    lip_roi = frame_gen[min_y:max_y, min_x:max_x]
    if lip_roi.size == 0:
        return 0
    
    # resize for syncnet
    lip_roi = cv2.resize(lip_roi, (96, 96))

    # 2. Prepare Input for SyncNet (convert to tensor, normalize, etc.):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for SyncNet
    ])
    lip_tensor = transform(lip_roi).unsqueeze(0).to(device)

    # 3. Get SyncNet Prediction:
    with torch.no_grad():
        syncnet_output = syncnet_model(lip_tensor)  # Requires correct input shape

    # 4. Calculate Loss (example - replace with appropriate loss function):
    target_label = torch.tensor([1.0], device=device)  # Example: Assume in-sync
    loss = F.binary_cross_entropy_with_logits(syncnet_output, target_label) # Example Loss

    return loss.item()
