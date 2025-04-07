from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import uuid
from inference.inference import GeneFace2Infer
import google.generativeai as genai
import logging
import re
import google.cloud.texttospeech as tts
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv()


app = FastAPI()

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
print(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Set up Google Generative AI
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

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

        # Extract text from the user's question
        text = data.question.strip()

        logging.info(f"Request received: {text}")

        # Use the LLM model to generate the initial response
        logging.info("Generating LLM response...")
        response = model.generate_content(data.question, generation_config={"max_output_tokens": 500})  # Allow more detail initially
        llm_output = remove_markdown(response.text)
        logging.info(f"LLM response generated: {llm_output}")

        # # Summarize the response to be concise and within 240 words
        # summary_prompt = f"Summarize the following response in a clear and concise manner while keeping it under 250 words:\n\n{llm_output}"
        # summary_response = model.generate_content(summary_prompt, generation_config={"max_output_tokens": 300})
        # concise_output = summary_response.text

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

    except Exception as e:
        logging.error(f"Error in /chat: {e}")  # Log the error
        return JSONResponse(status_code=500, content={"error": str(e)})
