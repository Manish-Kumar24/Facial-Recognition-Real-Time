import os
import cv2
from flask import Flask, render_template, jsonify
from keras.models import Model
from real_time_face_verification import capture_image, find_similar_matches
from utils.face_recognition import build_vgg_face_model, load_model_weights

app = Flask(__name__)
IMAGES_FOLDER = os.path.join(os.getcwd(), 'static', 'images')
MODEL_WEIGHTS_PATH = "models/vgg_face_weights.h5"

# Initialize VGG-Face descriptor
model = build_vgg_face_model()
load_model_weights(model, MODEL_WEIGHTS_PATH)
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture_and_verify():
    captured_image_path = os.path.join("static", "captured_image.jpg")

    # Start the camera and capture an image
    camera = cv2.VideoCapture(0)
    capture_image(camera, captured_image_path)
    camera.release()

    # Find similar matches
    try:
        # Use the updated find_similar_matches function
        similar_matches = find_similar_matches(vgg_face_descriptor, captured_image_path, IMAGES_FOLDER)

        if similar_matches:
            # Convert to web-friendly paths and format
            formatted_matches = []
            for match in similar_matches:
                matched_image_path = os.path.relpath(match['image_path'], os.getcwd())
                matched_image_path = matched_image_path.replace('\\', '/')
                
                formatted_matches.append({
                    "matched_image": matched_image_path,
                    "cosine_similarity": float(match['cosine_similarity']),
                    "euclidean_distance": float(match['euclidean_distance'])
                })

            response = {"matches": formatted_matches}
        else:
            response = {"error": "No similar faces found"}

    except Exception as e:
        response = {"error": str(e)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)