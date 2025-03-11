from albumentations import (
    ShiftScaleRotate, RandomBrightnessContrast, GaussianBlur, Resize, Compose
)
import cv2
import os

def get_aug_pipeline():
    """Define the Albumentations augmentation pipeline."""
    pipeline = Compose([
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=1.0),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        GaussianBlur(blur_limit=(3, 7), p=0.5)
        # Resize(height=512, width=512, p=1.0)  # Resize frames if needed
    ])
    return pipeline

def process_video_with_pipeline(input_video_path, output_video_path, pipeline):
    """
    Apply an Albumentations pipeline to a video.
    """
    # Read the video
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to RGB (Albumentations uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply Albumentations pipeline
        augmented = pipeline(image=frame_rgb)
        transformed_frame = augmented['image']

        # Convert back from RGB to BGR
        transformed_frame_bgr = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)

        # Write the transformed frame to the output video
        out.write(transformed_frame_bgr)
        frame_count += 1

        # Optional: Show progress
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    # Release resources
    cap.release()
    out.release()
    print(f"Video processed and saved to {output_video_path}")


# Example usage
pipeline = get_aug_pipeline()
input_video = "/home/arulmozg/GeneFacePlusPlus-LipSync/data/raw/videos/Jasper.mp4"  # Input video path
output_video = "/home/arulmozg/GeneFacePlusPlus-LipSync/data/raw/videos/augmented_video.mp4"  # Path to save the augmented video
process_video_with_pipeline(input_video, output_video, pipeline)
