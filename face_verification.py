import numpy as np
from utils.excel_utils import update_excel
from utils.preprocessing_image import preprocess_image 

# Threshold for cosine similarity
epsilon = 0.40  # Cosine similarity threshold

# Cosine Distance
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Euclidean Distance
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

# Verify Face
def verifyFace(vgg_face_descriptor, img1_path, img2_path):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1_path))[0, :]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2_path))[0, :]

    cosine_similarity = findCosineDistance(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

    print(f"Cosine Distance: {cosine_similarity}")
    print(f"Euclidean Distance: {euclidean_distance}")

    # Check if the faces are the same person based on cosine similarity
    if cosine_similarity < epsilon:
        same_person = "Verified: They are the same person."
        update_excel(img1_path, img2_path, cosine_similarity, euclidean_distance, same_person)
    else:
        same_person = "Unverified: They are not the same person."
    
    return cosine_similarity, euclidean_distance, same_person
