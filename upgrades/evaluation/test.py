import cv2
import torch
import lpips
import dlib
import numpy as np
from scipy.linalg import sqrtm
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

# Load LPIPS model once, outside the frame loop
def load_lpips_model(device):
    """Load the LPIPS model once."""
    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    return perceptual_loss

def calculate_lpips(frame_real, frame_gen, perceptual_loss, device):
    """Calculate LPIPS between two frames."""
    # Convert frames to tensors and normalize
    real_torch = torch.tensor(frame_real, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
    gen_torch = torch.tensor(frame_gen, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
    real_torch = real_torch.to(device)
    gen_torch = gen_torch.to(device)

    # No need to re-load the model each time
    with torch.no_grad():
        return perceptual_loss(real_torch, gen_torch).item()

# PSNR Calculation function (without the RGB conversion)
def calculate_psnr(frame_real, frame_gen):
    mse = np.mean((frame_real - frame_gen) ** 2)  # Mean Squared Error
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100  # PSNR formula
    return psnr


# Function to calculate Lip Movement Distance (LMD)
def calculate_lmd(frame_real, frame_gen, detector, predictor):
    """Calculate Lip Movement Distance (LMD) between two frames."""
    # Convert images to grayscale for face detection
    gray_real = cv2.cvtColor(frame_real, cv2.COLOR_BGR2GRAY)
    gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)

    # Detect faces in both images
    faces_real = detector(gray_real)
    faces_gen = detector(gray_gen)

    if len(faces_real) == 0 or len(faces_gen) == 0:
        return 0  # Return 0 if no faces are detected in either frame

    # Get landmarks for the first detected face in each frame
    landmarks_real = predictor(gray_real, faces_real[0])
    landmarks_gen = predictor(gray_gen, faces_gen[0])

    # Extract lip landmarks (landmarks 48-67)
    lip_landmarks_real = np.array([[p.x, p.y] for p in landmarks_real.parts()[49:68]])
    lip_landmarks_gen = np.array([[p.x, p.y] for p in landmarks_gen.parts()[49:68]])

    # Calculate Euclidean distance between corresponding lip landmarks
    lmd = np.mean(np.sqrt(np.sum((lip_landmarks_real - lip_landmarks_gen) ** 2, axis=1)))
    return lmd

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the LPIPS model
perceptual_loss = load_lpips_model(device)
# Load dlib's face landmark detector
predictor_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/test-scenarios/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load videos
real_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/data/raw/videos/May.mp4"
generated_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/generated-videos/may_infer.mp4"
predictor_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/test-scenarios/shape_predictor_68_face_landmarks.dat"  # Ensure this file is available

real_vid = cv2.VideoCapture(real_video_path)
gen_vid = cv2.VideoCapture(generated_video_path)

# Make sure the videos have the same FPS
fps_real = real_vid.get(cv2.CAP_PROP_FPS)
fps_gen = gen_vid.get(cv2.CAP_PROP_FPS)

if fps_real != fps_gen:
    print("Warning: The FPS of the videos do not match. Ensure the videos are synchronized.")

real_video_length = real_vid.get(cv2.CAP_PROP_FRAME_COUNT)  # Total frames in real video
generated_video_length = gen_vid.get(cv2.CAP_PROP_FRAME_COUNT)  # Total frames in generated video

print(real_video_length, generated_video_length)

min_frame_count = min(real_video_length, generated_video_length)

# Loop over the frames of the videos
lpips_scores = []
lmd_scores = []
psnr_scores = []

# Now, loop for the minimum number of frames to avoid length mismatch
for _ in range(int(min_frame_count)):
    ret_real, frame_real = real_vid.read()
    ret_gen, frame_gen = gen_vid.read()

    if not ret_real or not ret_gen:
        break  # Stop if we reach the end of either video

    # Calculate LPIPS score for the current frame pair
    lpips_score = calculate_lpips(frame_real, frame_gen, perceptual_loss, device)
    lpips_scores.append(lpips_score)
    lmd_score = calculate_lmd(frame_real, frame_gen, detector, predictor)
    lmd_scores.append(lmd_score)
    psnr_score = calculate_psnr(frame_real, frame_gen)
    psnr_scores.append(psnr_score)
    

# Release the video resources
real_vid.release()
gen_vid.release()

# Calculate the average LPIPS score
average_lpips_score = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
average_lmd_score = sum(lmd_scores) / len(lmd_scores) if lmd_scores else 0
average_psnr_score = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0


print(f"Average LPIPS score: {average_lpips_score:.4f}")
print(f"Average LMD score: {average_lmd_score:.4f}")
print(f"Average PSNR Score: {average_psnr_score:.4f} dB")
