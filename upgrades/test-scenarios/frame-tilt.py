import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

def apply_rotation_to_frame(frame, angle):
    """Apply rotation to a single frame."""
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, matrix, (w, h))
    return rotated_frame

def process_video(input_video_path, output_video_path, angle=30):
    """Process video with rotation and save the augmented video."""
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


        rotated_frame = apply_rotation_to_frame(frame, angle)
        out.write(rotated_frame)  # Writing rotated frame to the output video

    cap.release()
    out.release()
    print(f"Augmented video saved at {output_video_path}")

def process_audio(input_audio_path, output_audio_path):
    """Extract and save the original audio from the video."""
    video_clip = VideoFileClip(input_audio_path)
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
process_video(input_video_path, output_video_path, angle=30)

input_audio_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/preprocess/May.mp4"
output_audio_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/preprocess/original_audio.wav"
process_audio(input_audio_path, output_audio_path)

final_output_path = "/home/arulmozg/GeneFacePlusPlus-LipSync/preprocess/dataset/tilted_may_video.mp4"
combine_audio_video(output_video_path, output_audio_path, final_output_path)
