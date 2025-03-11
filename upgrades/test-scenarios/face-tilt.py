import cv2
import dlib
import numpy as np
import os

# Load face detector and landmark predictor
# At the start of your script

detector = dlib.get_frontal_face_detector()
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Missing {shape_predictor_path}. Please download and place it in the working directory.")

predictor = dlib.shape_predictor(shape_predictor_path)

def get_face_angle(landmarks):
    """Estimate face angle based on eye positions."""
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    return np.degrees(np.arctan2(dy, dx))


def rotate_face(image, rect, angle):
    """Rotate the detected face region and blend it seamlessly."""
    (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
    center = (x + w // 2, y + h // 2)
    
    # Extract face region
    face_roi = image[y:y+h, x:x+w].copy()
    
    # Compute rotation matrix
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    
    # Rotate the face ROI
    rotated_face = cv2.warpAffine(face_roi, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Create a mask for seamless blending
    mask = np.full((h, w), 255, dtype=np.uint8)
    
    # Blend the rotated face back using seamlessClone
    image[y:y+h, x:x+w] = cv2.seamlessClone(rotated_face, image[y:y+h, x:x+w], mask, (w//2, h//2), cv2.NORMAL_CLONE)
    
    return image


def process_video(input_video_path, output_video_path, desired_tilt):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_video_path}")
    
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Default to 30 if FPS is 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_height, frame_width, fps, total_frames)
    
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if faces:
            for face in faces:
                landmarks = predictor(gray, face)
                face_angle = get_face_angle(landmarks)
                angle_adjustment = desired_tilt - face_angle
                frame = rotate_face(frame, face, angle_adjustment)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    print("Video processing completed!")
    print(f"Processed video saved at {output_video_path}")


# if __name__ == "__main__":
input_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/May.mp4"
output_video_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/upgrades/dataset/augmented_video.mp4"


process_video(input_video_path=input_video_path, output_video_path=output_video_path, desired_tilt=15)

