VisionAI: AI-Powered Navigation Assistant
Empowering the Visually Impaired through Real-Time Computer Vision and Intelligent Voice Guidance.

VisionAI is a sophisticated assistive technology platform designed to bridge the gap between complex environments and safe mobility. By combining high-speed object detection with natural language feedback, it serves as a "digital eye" for users, providing them with a clear, narrated understanding of their surroundings.

Core Project Vision
The primary goal of VisionAI is to provide spatial awareness. Unlike standard GPS which only gives location, VisionAI provides context—identifying obstacles, recognizing people, and interpreting the immediate path ahead to prevent accidents before they happen.

Advanced Features
1. Real-Time Spatial Awareness
Using the YOLOv8 (You Only Look Once) architecture, the system performs sub-second inference on live video frames. It doesn't just name objects; it calculates their relative position to the user.

Proximity Detection: Estimates distance based on bounding box scaling.

Dynamic Tracking: Follows moving objects (like cars or pedestrians) to predict potential collisions.

2. Intelligent Voice Interface
The system features a dual-mode audio system:

Active Narration: Automatically announces critical obstacles in the user's path.

Interactive Querying: Powered by OpenAI Whisper, users can ask specific questions like "Is there a chair nearby?" or "Where is the exit?" and receive immediate verbal answers.

3. Low-Latency Processing Engine
To ensure safety, the "Glass-to-Ear" latency (the time from a camera capturing a frame to the user hearing a warning) is minimized through:

Asynchronous Processing: FastAPI handles frame ingestion and AI inference on separate threads.

WebSocket Streams: Persistent bi-directional connections ensure data flows without the overhead of traditional HTTP requests.

System Architecture & Logic
The project is built on a "Triangle of Intelligence" model:

Perception Layer (YOLOv8 & OpenCV): Converts raw pixels into structured data (Labels, Confidence Scores, and Coordinates).

Logic Layer (FastAPI): Filters the data to prioritize "Critical Obstacles" (e.g., a staircase is more important than a wall).

Interaction Layer (React & Web Speech API): Converts the prioritized data into natural, human-like speech for the user.
