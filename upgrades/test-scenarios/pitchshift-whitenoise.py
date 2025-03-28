from pydub import AudioSegment
from pydub.generators import WhiteNoise
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np

def augment_audio(input_audio_path, output_audio_path, pitch_shift_semitones=2, noise_level=0.02):
    """Augment audio by applying pitch shifting and adding noise."""
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)

    # Apply pitch shifting
    new_sample_rate = int(audio.frame_rate * (2 ** (pitch_shift_semitones / 12.0)))
    pitched_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate}).set_frame_rate(audio.frame_rate)

    # Generate and add white noise
    noise = WhiteNoise().to_audio_segment(duration=len(pitched_audio), volume=-40)  # Adjust noise volume
    noisy_audio = pitched_audio.overlay(noise, gain_during_overlay=noise_level)

    # Export the augmented audio
    noisy_audio.export(output_audio_path, format="mp3")
    print(f"Augmented audio saved at {output_audio_path}")

def combine_audio_video(input_video_path, augmented_audio_path, output_video_path):
    """Combine augmented audio with the original video."""
    # Load video
    video = VideoFileClip(input_video_path)

    # Load augmented audio
    audio = AudioFileClip(augmented_audio_path)

    # Combine video and audio
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    print(f"Final video saved at {output_video_path}")

# Define paths
input_video_path = "path/to/your/input_video.mp4"  # Update with your actual video path
input_audio_path = "path/to/your/input_audio.mp3"  # Extract the audio track from the video if needed
augmented_audio_path = "path/to/your/augmented_audio.mp3"
output_video_path = "path/to/your/output_video.mp4"

# Step 1: Augment the audio
augment_audio(input_audio_path, augmented_audio_path, pitch_shift_semitones=2, noise_level=0.02)

# Step 2: Combine the augmented audio with the original video
combine_audio_video(input_video_path, augmented_audio_path, output_video_path)
