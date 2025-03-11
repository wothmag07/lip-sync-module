ffmpeg -i May.mp4 -q:a 0 -map a output_audio.wav
ffmpeg -i augmented_video.mp4 -i original_audio.wav -c:v copy -c:a aac output.mp4
