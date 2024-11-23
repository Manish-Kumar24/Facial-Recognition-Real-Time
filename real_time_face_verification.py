import os
import cv2
import numpy as np
from face_verification import verifyFace
from utils.preprocessing_image import preprocess_image
from utils.excel_utils import update_excel
from utils.face_recognition import build_vgg_face_model, load_model_weights
from keras.models import Model

# Path to the saved images
IMAGES_FOLDER = "images"
MODEL_WEIGHTS_PATH = "models/vgg_face_weights.h5"
SIMILARITY_THRESHOLD = 0.5  # Similarity threshold for matching

def capture_image(camera, save_path):
    """
    Captures an image from the camera after 5 seconds.
    """
    print("Starting camera... Get ready!")
    cv2.namedWindow("Live Feed")
    start_time = cv2.getTickCount()
    fps = cv2.getTickFrequency()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break

        elapsed_time = (cv2.getTickCount() - start_time) / fps
        if elapsed_time >= 5:
            cv2.imwrite(save_path, frame)
            print(f"Image captured and saved at {save_path}")
            break

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit early if needed
            break

    cv2.destroyAllWindows()

def find_similar_matches(vgg_face_descriptor, captured_image_path, images_folder):
    similar_matches = []

    for image_file in os.listdir(images_folder):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_folder, image_file)
            cosine_similarity, euclidean_distance, same_person = verifyFace(vgg_face_descriptor, captured_image_path, image_path)

            # Store matches above similarity threshold
            if cosine_similarity <= SIMILARITY_THRESHOLD:
                similar_matches.append({
                    'image_file': image_file,
                    'image_path': image_path, 
                    'cosine_similarity': cosine_similarity, 
                    'euclidean_distance': euclidean_distance
                })

                # Update Excel for each similar match
                update_excel(captured_image_path, image_path, cosine_similarity, euclidean_distance, "Verified")

    # Sort matches by similarity (lower is better for cosine similarity)
    similar_matches.sort(key=lambda x: x['cosine_similarity'])

    return similar_matches

def main():
    # Initialize the VGG-Face model
    model = build_vgg_face_model()
    load_model_weights(model, MODEL_WEIGHTS_PATH)
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    camera = cv2.VideoCapture(0)
    captured_image_path = "captured_image.jpg"
    capture_image(camera, captured_image_path)
    camera.release()

    similar_matches = find_similar_matches(vgg_face_descriptor, captured_image_path, IMAGES_FOLDER)

    if similar_matches:
        for match in similar_matches:
            print(f"Matched Image: {match['image_file']}")
            print(f"Cosine Similarity: {match['cosine_similarity']}")
            print(f"Euclidean Distance: {match['euclidean_distance']}")

        # Display the captured and matched images
        captured_img = cv2.imread(captured_image_path)
        
        # Display multiple matched images
        for match in similar_matches:
            matched_img = cv2.imread(match['image_path'])
            combined = np.hstack((cv2.resize(captured_img, (400, 400)), cv2.resize(matched_img, (400, 400))))
            cv2.imshow(f"Captured vs {match['image_file']}", combined)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
    else:
        print("No match found.")

if __name__ == "__main__":
    main()