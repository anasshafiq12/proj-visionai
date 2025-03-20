import cv2
import numpy as np
import threading
import queue
import base64
from flask import Flask, request, jsonify
from object_detection import detect_objects
from scene_description import describe_scene
from audio_feedback import generate_audio
from gps_integration import process_gps_data
from audio_input import listen
from conversation import process_voice_command
from navigation import generate_navigation_instructions as get_navigation_instructions
from flask import Flask, send_from_directory
import time

# Initialize the Flask app
app = Flask(__name__)

# Queue for incoming frames with a max size to prevent memory issues
frame_queue = queue.Queue(maxsize=10)

# Thread-safe last navigation instruction
last_navigation_text = None
lock = threading.Lock()  # Ensures thread-safe access to `last_navigation_text`


# Thread-safe functions to manage the `last_navigation_text`
def update_navigation_text(new_text):
    global last_navigation_text
    with lock:
        last_navigation_text = new_text


def get_navigation_text():
    global last_navigation_text
    with lock:
        return last_navigation_text


# Function to process incoming frames
def process_frame():
    while True:
        try:
            if not frame_queue.empty():
                frame_data = frame_queue.get()
                
                # Decode Base64 string to bytes
                frame_bytes = base64.b64decode(frame_data)
                np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("Error: Could not decode image")
                    continue

                # Object Detection
                detected_objects = detect_objects(frame)

                # Generate Navigation Guidance
                navigation_text = get_navigation_instructions(detected_objects)

                # Skip redundant navigation instructions
                if navigation_text == get_navigation_text():
                    print("Skipping redundant navigation instruction:", navigation_text)
                    frame_queue.task_done()
                    continue

                # Update the last navigation text
                update_navigation_text(navigation_text)

                # Log the new instruction
                print("Processed Frame:", {
                    "navigation_instruction": navigation_text
                })

                frame_queue.task_done()  # Mark task as complete
            
            # Add sleep to process frames every 10 seconds
            time.sleep(10)
        except Exception as e:
            print(f"Error processing frame: {e}")


# Start the frame processing thread
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Receives Base64-encoded video frames from the Android app and returns navigation instructions."""
    try:
        data = request.get_json()
        if not data or "frame" not in data:
            return jsonify({"error": "No frame provided"}), 400

        # Decode Base64 frame data
        frame_data = data["frame"]
        frame_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Process the frame
        detected_objects = detect_objects(frame)
        navigation_text = get_navigation_instructions(detected_objects)

        # Skip redundant navigation instructions
        if navigation_text == get_navigation_text():
            return jsonify({
                "status": "No new navigation instruction"
            }), 200

        # Update the last navigation instruction
        update_navigation_text(navigation_text)

        # Return the new instruction
        return jsonify({
            "status": "Frame processed successfully",
            "navigation_instruction": navigation_text
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process frame: {str(e)}"}), 500


@app.route('/gps_data', methods=['POST'])
def gps_data():
    """Receives GPS coordinates from the Android app."""
    try:
        gps_info = request.get_json()
        if not gps_info or "latitude" not in gps_info or "longitude" not in gps_info:
            return jsonify({"error": "Invalid GPS data"}), 400

        processed_info = process_gps_data(gps_info)
        return jsonify({"status": "GPS data processed", "message": processed_info}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process GPS data: {str(e)}"}), 500


@app.route("/voice_command", methods=["POST"])
def voice_command():
    """Processes voice commands from the Android app."""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "No voice command provided"}), 400

        query = data["query"]
        response = process_voice_command(query)  # Adjust to process query string
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process voice command: {str(e)}"}), 500


@app.route('/audio_responses/<filename>')
def serve_audio(filename):
    """Serves pre-generated audio files."""
    return send_from_directory('audio_responses', filename)


if __name__ == "__main__":
    print("Starting the Blind Navigation AI server...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)  # Threaded for better concurrency
