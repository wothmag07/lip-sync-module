import google.cloud.texttospeech as tts
import os
from dotenv import load_dotenv

load_dotenv()

def read_text_from_file(file_path: str) -> str:
    """Reads text from a given file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def text_to_wav(language_code: str, voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    filename = f"{voice_name}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')
        

if __name__ == "__main__":
    inputtxt = read_text_from_file("data/raw/text-files/input.txt")
    text_to_wav(language_code='en-US', voice_name="en-US-Standard-E", text=inputtxt)

    

    