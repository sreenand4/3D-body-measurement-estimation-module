from Scanner import Scanner
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import psutil
import gc

app = Flask(__name__)
CORS(app)
scanner = Scanner()  # Global instance

def decode_base64_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]  # Remove data URL prefix if present
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def check_landmark_quality(image, landmarks, view_name):
    mp_pose = scanner.mp_pose  # Use global scanner
    height, width = image.shape[:2]
    
    required_landmarks = {
        "front": [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.RIGHT_PINKY,
            mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX,
        ],
        "back": [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_THUMB, mp_pose.PoseLandmark.RIGHT_THUMB,
            mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL
        ],
        "side": [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
        ]
    }
    
    if not landmarks:
        return False, f"No landmarks detected for {view_name} view"
    
    failed_landmarks = []
    for landmark in required_landmarks[view_name]:
        visibility = landmarks.landmark[landmark].visibility
        if visibility is None or visibility < 0.15:
            failed_landmarks.append((landmark.name.lower().replace("_", "-"), visibility if visibility is not None else "None"))
    
    if failed_landmarks:
        # Join just the landmark names, not the tuples
        failed_landmark_names = [x[0] for x in failed_landmarks]
        return False, f"Quality check failed for {view_name} view. Please re-take the image and make sure {', '.join(failed_landmark_names)} are visible."
    
    return True, "Quality check passed"

@app.route('/estimate', methods=['POST'])
def estimate_measurements():
    try:
        data = request.get_json()
        if not data or not all(key in data for key in ["front", "side", "back"]) or 'height' not in data:
            return jsonify({"error": "Missing or invalid data: expected {'front': '...', 'side': '...', 'back': '...', 'height': ...}"}), 400

        height = float(data['height'])
        images = {
            "front": decode_base64_image(data["front"]),
            "side": decode_base64_image(data["side"]),
            "back": decode_base64_image(data["back"])
        }
        gender = data["gender"]

        # Check image decoding
        if any(img is None for img in images.values()):
            return jsonify({"error": "Failed to decode one or more images"}), 400

        # Log initial memory usage
        process = psutil.Process(os.getpid())
        print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024} MB")

        # Process images to get landmarks and create visualizations
        landmark_checks = {}
        visualizations = {}
        for view, image in images.items():
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = scanner.pose.process(image_rgb)
            landmark_checks[view] = check_landmark_quality(image, pose_results.pose_landmarks, view)

            # Create visualization with landmarks overlaid
            annotated_image = image_rgb.copy()
            if pose_results.pose_landmarks:
                scanner.mp_drawing.draw_landmarks(
                    annotated_image,
                    pose_results.pose_landmarks,
                    scanner.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=scanner.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            # Convert back to BGR for consistency with original image
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            visualizations[view] = encode_image_to_base64(annotated_image)

            if not landmark_checks[view][0]:  # If quality check fails
                return jsonify({
                    "error": landmark_checks[view][1],
                    "visualizations": visualizations
                }), 400

        # If all quality checks pass, proceed with measurement calculation
        measurements = scanner.process_images(images["front"], images["side"], images["back"], height, gender)

        # Log final memory usage
        print(f"Final memory usage: {process.memory_info().rss / 1024 / 1024} MB")
        gc.collect()

        return jsonify({
            "status": "success",
            "measurements": measurements,
            "visualizations": visualizations
        }), 200
    

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)