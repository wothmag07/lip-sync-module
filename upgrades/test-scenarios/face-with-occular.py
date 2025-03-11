import cv2
import numpy as np
import os
import dlib

detector = dlib.get_frontal_face_detector()
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Missing {shape_predictor_path}. Please download and place it in the working directory.")

predictor = dlib.shape_predictor(shape_predictor_path)

def overlay_image(background, overlay, x, y, w, h):
    """
    Overlays an image (with transparency) onto another image.
    """
    overlay = cv2.resize(overlay, (w, h))
    for i in range(h):
        for j in range(w):
            if overlay[i, j][3] != 0:  # If pixel is not transparent
                background[y + i, x + j] = overlay[i, j][:3]
    return background

def add_occlusion(frame, occlusion_type="mask"):
    """
    Applies a realistic occlusion (mask, glasses) using preloaded images.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    mask_img = cv2.imread("/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/Surgicalmask.png", cv2.IMREAD_UNCHANGED)
    glasses_img = cv2.imread("/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/glasses.png", cv2.IMREAD_UNCHANGED)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        if occlusion_type == "mask":
            x1, y1 = landmarks.part(2).x - 25, landmarks.part(29).y  # Left cheek to nose tip
            x2, y2 = landmarks.part(14).x + 5, landmarks.part(33).y  # Right cheek to chin

            mask_width = (x2 - x1) * 1.3  # Increase width by 20%
            mask_height = (y2 - y1) * 3.5  # Increase height by 30%

            frame = overlay_image(frame, mask_img, x1, y1, int(mask_width), int(mask_height))

        elif occlusion_type == "glasses":
            x1, y1 = landmarks.part(36).x - 25, landmarks.part(23).y  # Adjust left eye & top eyebrow
            x2, y2 = landmarks.part(45).x - 10, landmarks.part(30).y  # Adjust right eye & nose bridge

            glasses_width = (x2 - x1) * 1.2  # Increase width by 20%
            glasses_height = (y2 - y1) * 1.2  # Increase height by 20%

            frame = overlay_image(frame, glasses_img, x1, y1, int(glasses_width), int(glasses_height))

    
    return frame

def process_video(input_video_path, output_video_path, occlusion_type):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_video_path}")
    
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Properties - Height: {frame_height}, Width: {frame_width}, FPS: {fps}, Total Frames: {total_frames}")
    
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = add_occlusion(frame, occlusion_type)
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    print("Video processing completed!")
    print(f"Processed video saved at {output_video_path}")

# Example Usage
# process_video("input.mp4", "output_occluded.mp4", "mask")

input_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/May.mp4"
# output_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/face_occular_glasses_video.mp4"
output_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/face_occular_mask_video.mp4"



process_video(input_video_path=input_video_path, output_video_path=output_video_path, occlusion_type="mask")