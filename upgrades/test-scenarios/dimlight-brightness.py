import cv2
import numpy as np
import os

def adjust_brightness_contrast(frame, alpha=1.0, beta=0):
    """
    Adjust brightness and contrast of a frame.
    alpha > 1 → Increase contrast
    beta > 0 → Increase brightness
    """
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return adjusted

def process_video(input_video_path, output_video_path, effect="dim"):
    """Process video with lighting effects (No face tilting)."""
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply lighting effects
        if effect == "dim":
            processed_frame = adjust_brightness_contrast(frame, alpha=0.7, beta=-50)  # Darker
        elif effect == "overexposed":
            processed_frame = adjust_brightness_contrast(frame, alpha=1.5, beta=80)  # Brighter

        out.write(processed_frame)

    cap.release()
    out.release()
    print(f"Processed video with {effect} lighting saved at {output_video_path}")


# File paths
input_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/May.mp4"
output_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/overexposed_augmented_video.mp4"


# Processing with dim lighting
process_video(input_video_path, output_video_path, effect="overexposed")

