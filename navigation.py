# Store the last spoken instruction globally
last_instruction = None  

def generate_navigation_instructions(detected_objects):
    """
    Generates movement instructions based on detected objects.

    Args:
        detected_objects (list): A list of detected objects with labels and bounding boxes.

    Returns:
        str: Navigation instructions for the user, or None if it's the same as the last instruction.
    """
    global last_instruction

    if not detected_objects:
        instruction = "No obstacles detected. You can move forward safely."
    else:
        obstacles = []
        open_path = True

        for obj in detected_objects:
            label = obj["label"]
            x1, _, x2, _ = obj["bbox"]  # Extract bounding box (x-coordinates)
            center_x = (x1 + x2) / 2  # Get center position

            # Define general zones based on frame width (assuming 640px)
            if center_x < 213:  
                position = "on your left"
            elif center_x > 426:  
                position = "on your right"
            else:  
                position = "in front of you"
                open_path = False  # Blocked path

            obstacles.append(f"a {label} {position}")

        # Construct navigation instructions
        if open_path:
            instruction = "There are obstacles around, but the path ahead is clear. Move forward carefully."
        else:
            instruction = f"You should stop. There is {', '.join(obstacles)}. Try moving slightly left or right."

    # Check if this instruction is the same as the last one
    if instruction == last_instruction:
        return None  # Don't repeat the same instruction

    last_instruction = instruction  # Update last instruction
    return instruction  # Return new instruction only

# Test code
if __name__ == "__main__":
    sample_objects = [
        {"label": "chair", "bbox": [100, 200, 250, 400]},  # Left side
        {"label": "table", "bbox": [300, 100, 500, 350]},  # Center
        {"label": "door", "bbox": [500, 50, 600, 200]}  # Right side
    ]
    print(generate_navigation_instructions(sample_objects))  # Should print instruction
    print(generate_navigation_instructions(sample_objects))  # Should print None (no change)
