from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import uuid
import glob  # Import glob for file matching
from inference.inference import GeneFace2Infer
import google.generativeai as genai
import logging
import re
import google.cloud.texttospeech as tts
from dotenv import load_dotenv
from fastapi import HTTPException, status
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.generativeai.types.generation_types import StopCandidateException
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv()


app = FastAPI()

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
print(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

def cleanup_old_videos():
    logging.info(f"Cleaning up old video files in {static_dir}...")
    video_files = glob.glob(os.path.join(static_dir, "output_*.mp4"))
    for f in video_files:
        try:
            os.remove(f)
            logging.info(f"Deleted old video file: {f}")
        except OSError as e:
            logging.error(f"Error deleting file {f}: {e}")

# Startup event handler to clean up old videos
@app.on_event("startup")
async def cleanup_videos():
    logging.info(f"Cleaning up old video files in {static_dir}...")
    video_files = glob.glob(os.path.join(static_dir, "output_*.mp4"))
    deleted_count = 0
    for f in video_files:
        try:
            os.remove(f)
            deleted_count += 1
            logging.info(f"Deleted old video file: {f}")
        except OSError as e:
            logging.error(f"Error deleting file {f}: {e}")
    logging.info(f"Cleanup complete. Deleted {deleted_count} video files.")


# Initialize Gemini AI Model
model = genai.GenerativeModel("gemini-1.5-pro")

def remove_markdown(text):
    return re.sub(r"[*_`~<>\[\]]", "", text).strip()

# Define request model
class ChatRequest(BaseModel):
    question: str
    # voice: str = "female"  # Default to male voice


# Initialize GeneFace2Infer instance
infer_instance = GeneFace2Infer(
    audio2secc_dir="checkpoints/audio2motion_vae",
    postnet_dir="",
    head_model_dir="checkpoints/motion2video_nerf/May_head",
    torso_model_dir="checkpoints/motion2video_nerf/May_torso",
)

# Serve frontend HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request, data: ChatRequest):
    try:
        request_id = str(uuid.uuid4())
        video_url = ""

        cleanup_old_videos()  # Clean up old videos on each request
        # Extract text from the user's question
        text = data.question.strip()

        logging.info(f"Request received: {text}")

        # Use the LLM model to generate the initial response
        logging.info("Generating LLM response...")
        response = model.generate_content(data.question, generation_config={"max_output_tokens": 100})  # Allow more detail initially
        llm_output = remove_markdown(response.text)
        logging.info(f"LLM response generated (initial): {llm_output}")
        
        # Check word count and summarize if necessary
        word_count = len(llm_output.split())
        if word_count > 150:
            logging.info(f"LLM response exceeds 50 words ({word_count}). Summarizing...")
            summary_prompt = f"Summarize the following text to be under 50 words, keeping the core message:\n\n{llm_output}"
            try:
                # Use a slightly higher max_output_tokens for summarization buffer
                summary_response = model.generate_content(summary_prompt, generation_config={"max_output_tokens": 75}) 
                summarized_text = remove_markdown(summary_response.text)
                # Double-check summarized word count
                if len(summarized_text.split()) <= 50:
                     llm_output = summarized_text
                     logging.info(f"Summarized LLM response: {llm_output}")
                else:
                     logging.warning(f"Summarization still exceeded 50 words ({len(summarized_text.split())}). Using truncated original response.")
                     # Fallback: Truncate original response if summary is still too long
                     llm_output = " ".join(llm_output.split()[:50]) + "..." 
            except Exception as summary_e:
                logging.error(f"Error during summarization: {summary_e}. Using truncated original response.")
                # Fallback: Truncate original response if summarization fails
                llm_output = " ".join(llm_output.split()[:50]) + "..."

        # Prepare the input for the GeneFace2Infer model
        logging.info("Preparing input for GeneFace2Infer...")
        inp = {
            'a2m_ckpt': "checkpoints/audio2motion_vae",
            'postnet_ckpt': "",
            'head_ckpt': "checkpoints/motion2video_nerf/May_head",
            'torso_ckpt': "checkpoints/motion2video_nerf/May_torso",
            'input_txt': llm_output,  # Pass the text directly
            # 'voice': data.voice,  # Use the voice from the request
            'voice': 'female',  # Always female
            'drv_pose': 'static',
            'blink_mode': 'period',
            'temperature': 0.2,
            'mouth_amp': 0.4,
            'lle_percent': 0.2,
            'debug': False,
            'out_name': os.path.join(static_dir, f"output_{request_id}.mp4"),
            'raymarching_end_threshold': 0.01,
            'low_memory_usage': True,
        }
        logging.info("Input prepared.")

        # Generate the video using the GeneFace2Infer model
        logging.info("Generating video using GeneFace2Infer...")
        video_file_name = infer_instance.infer_once(inp)
        logging.info(f"Video file generated: {video_file_name}")
        video_url = f"/static/{video_file_name}"

        return {"answer": llm_output, "videoUrl": video_url}

    except StopCandidateException as ge:
        logging.error(f"Gemini Error: {ge}")
        status_code = getattr(ge, "status_code", 400)
        return JSONResponse(status_code=status_code, content={"error": str(ge)})

    except GoogleAPIError as gerr:
        logging.error(f"Google API Error: {gerr}")
        status_code = getattr(gerr, "code", 400)
        return JSONResponse(status_code=status_code, content={"error": str(gerr)})

    except DefaultCredentialsError as dce:
        logging.error(f"Credentials Error: {dce}")
        return JSONResponse(status_code=401, content={"error": "Missing or invalid Google credentials."})

    except HTTPException as http_exc:
        logging.error(f"HTTP Exception: {http_exc}")
        return JSONResponse(status_code=http_exc.status_code, content={"error": http_exc.detail})

    except Exception as e:
        logging.error("Unhandled exception in /chat")
        traceback.print_exc()  # optional, for full traceback in logs

        # Extract common status-like attributes
        status_code = (
            getattr(e, "status_code", None) or
            getattr(e, "code", None) or
            getattr(e, "response", {}).get("status", None)
        )

        # Ensure it’s a valid int, else fallback
        status_code = status_code if isinstance(status_code, int) else 500

        return JSONResponse(
            status_code=status_code,
            content={"error": str(e)}
        )
