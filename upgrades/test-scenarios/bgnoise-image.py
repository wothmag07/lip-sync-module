import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

def add_noise_to_frame(frame, noise_factor=0.05):
    """Add random noise to a frame."""
    row, col, ch = frame.shape
    mean = 0
    sigma = noise_factor * 255
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_frame = np.clip(frame + gauss, 0, 255).astype(np.uint8)
    return noisy_frame

def process_video_with_noise(input_video_path, output_video_path, noise_factor=0.05):
    """Process video by adding noise and save the augmented video."""
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():  # Check if the video file is opened successfully
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read a frame from the video.")
            break
        
        # Add noise to the frame
        noisy_frame = add_noise_to_frame(frame, noise_factor)
        out.write(noisy_frame)  # Write noisy frame to the output video

    cap.release()
    out.release()
    print(f"Augmented video saved at {output_video_path}")


def process_audio(input_video_path, output_audio_path):
    """Extract and save the original audio from the video."""
    video_clip = VideoFileClip(input_video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)
    print(f"Audio saved at {output_audio_path}")

def combine_audio_video(video_path, audio_path, output_path):
    """Combine the augmented video and audio."""
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    video_with_audio = video_clip.set_audio(audio_clip)

    # Write the final video to disk
    video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"Final video saved at {output_path}")


input_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/preprocess/May.mp4"
output_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/preprocess/augmented_video.mp4"
process_video_with_noise(input_video_path, output_video_path, noise_factor=0.05)

output_audio_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/preprocess/original_audio.wav"
process_audio(input_video_path, output_audio_path)

final_output_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/preprocess/dataset/bgnoise_may.mp4"
combine_audio_video(output_video_path, output_audio_path, final_output_path)
