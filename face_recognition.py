import numpy as np
from keras.models import Sequential, Model
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dropout, Activation
from face_verification import verifyFace
from utils.preprocessing_image import preprocess_image
from utils.excel_utils import update_excel

# Define the VGG-Face model
def build_vgg_face_model():
    model = Sequential()

    # Block 1
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully connected layers
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model


# Load the pre-trained weights
def load_model_weights(model, weights_path):
    try:
        model.load_weights(weights_path)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit(1)


# Main function to execute face verification
def main():
    # Initialize and load model
    model = build_vgg_face_model()
    load_model_weights(model, 'models/vgg_face_weights.h5')

    # Define VGG-Face descriptor
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    # Test face verification
    img1_path = 'images/MANISH_GRAY.jpg'
    img2_path = 'images/MANISH_RGB.jpg'
    
    # Call verifyFace and capture result
    cosine_sim, euclidean_dist, same_person = verifyFace(vgg_face_descriptor, img1_path, img2_path)
    
    print(f"Cosine Similarity: {cosine_sim}")
    print(f"Euclidean Distance: {euclidean_dist}")
    # print(f"Are the faces the same person? {same_person}")
    print(same_person)

    # Update the Excel sheet with results
    update_excel(img1_path, img2_path, cosine_sim, euclidean_dist, same_person)

if __name__ == "__main__":
    main()
