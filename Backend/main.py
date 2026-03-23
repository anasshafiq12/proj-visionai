import cv2
import numpy as np
import asyncio
import base64
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Original Module Imports
from object_detection import detect_objects
from scene_description import describe_scene
from audio_feedback import generate_audio
from gps_integration import process_gps_data
from audio_input import listen
from conversation import process_voice_command
from navigation import generate_navigation_instructions as get_navigation_instructions

app = FastAPI()

# 1. Thread-safe/Async-safe state management
class NavigationState:
    def __init__(self):
        self.last_text = None
        self.lock = asyncio.Lock()
        # To replicate your queue.Queue(maxsize=10)
        self.frame_queue = asyncio.Queue(maxsize=10)

    async def update(self, text):
        async with self.lock:
            self.last_text = text

    async def get(self):
        async with self.lock:
            return self.last_text

nav_state = NavigationState()

# 2. Static File Serving (Replaces serve_audio)
if not os.path.exists('audio_responses'):
    os.makedirs('audio_responses')
app.mount("/audio_responses", StaticFiles(directory="audio_responses"), name="audio_responses")

# 3. Background Process (Replicates your process_frame thread)
async def process_frame_task():
    """Background loop that processes frames every 10 seconds if the queue isn't empty."""
    while True:
        try:
            if not nav_state.frame_queue.empty():
                frame_data = await nav_state.frame_queue.get()
                
                # Decode Base64 string to bytes
                frame_bytes = base64.b64decode(frame_data)
                np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("Error: Could not decode image")
                    nav_state.frame_queue.task_done()
                    continue

                # Object Detection
                detected_objects = detect_objects(frame)

                # Generate Navigation Guidance
                navigation_text = get_navigation_instructions(detected_objects)

                # Skip redundant navigation instructions
                last_text = await nav_state.get()
                if navigation_text == last_text:
                    print("Skipping redundant navigation instruction:", navigation_text)
                else:
                    await nav_state.update(navigation_text)
                    print(f"Processed Frame: {{'navigation_instruction': {navigation_text}}}")

                nav_state.frame_queue.task_done()
            
            # Replicates your time.sleep(10) without blocking the server
            await asyncio.sleep(10)
        except Exception as e:
            print(f"Error processing frame: {e}")

# Start the background task on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_frame_task())

# 4. API Endpoints

@app.post("/upload_frame")
async def upload_frame(request: Request):
    """Receives frames and processes them immediately (as per your Flask route)."""
    try:
        data = await request.json()
        if not data or "frame" not in data:
            raise HTTPException(status_code=400, detail="No frame provided")

        frame_data = data["frame"]
        
        # Add to queue for the background loop (if you still want that logic)
        if not nav_state.frame_queue.full():
            await nav_state.frame_queue.put(frame_data)

        # Immediate processing for the HTTP response
        frame_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        detected_objects = detect_objects(frame)
        navigation_text = get_navigation_instructions(detected_objects)

        last_text = await nav_state.get()
        if navigation_text == last_text:
            return {"status": "No new navigation instruction"}

        await nav_state.update(navigation_text)

        return {
            "status": "Frame processed successfully",
            "navigation_instruction": navigation_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gps_data")
async def gps_data(request: Request):
    try:
        gps_info = await request.json()
        if not gps_info or "latitude" not in gps_info or "longitude" not in gps_info:
            raise HTTPException(status_code=400, detail="Invalid GPS data")

        processed_info = process_gps_data(gps_info)
        return {"status": "GPS data processed", "message": processed_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_command")
async def voice_command(request: Request):
    try:
        data = await request.json()
        if not data or "query" not in data:
            raise HTTPException(status_code=400, detail="No voice command provided")

        query = data["query"]
        response = process_voice_command(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Running the app
if __name__ == "__main__":
    import uvicorn
    print("Starting the Blind Navigation AI server...")
    uvicorn.run(app, host="0.0.0.0", port=5000)