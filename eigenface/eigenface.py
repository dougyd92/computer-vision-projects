# Doug de Jesus
# drd8913@nyu.edu
# N14928011

import os
import numpy as np
from PIL import Image


# Todo want a way to have code reuse
# but also still access width, height, and filenames
# def load_images(directory):
#     images = []
#     files = os.listdir(directory)
#     for filename in files:
#         img_path = os.path.join(directory, filename)
#         img = Image.open(img_path)
#         images.append(img)
#     return images

# def images_to_col_vectors(images_arr):
#     image_vectors = []
#     for img in images_arr:
#         column_vector = np.array(img.getdata())[:, np.newaxis]
#         image_vectors.append(column_vector)
#     return image_vectors


def output_image(gray_values, height, width, directory, filename):
    pixels = np.uint8(np.reshape(gray_values, (height, width)))
    img = Image.fromarray(pixels, mode='L')
    img_path = os.path.join(directory, filename)
    img.save(img_path)


def main():
    # todo argparse
    training_dir = "../Project2/Face dataset/Training"
    test_dir = "../Project2/Face dataset/Testing"
    output_dir = "../Project2/Face dataset/Output"

    # training_images = load_images(training_dir)
    # training_image_vectors = images_to_col_vectors(training_images) # todo see above
    training_files = os.listdir(training_dir)
    training_images = []
    for filename in training_files:
        img_path = os.path.join(training_dir, filename)
        img = Image.open(img_path)
        training_images.append(img)

    M = len(training_images)
    height = training_images[0].height
    width = training_images[0].width

    training_image_vectors = [] # todo remove this
    for img in training_images:
        column_vector = np.array(img.getdata())[:, np.newaxis]
        training_image_vectors.append(column_vector)

    # This will have dimension (h*w) x M
    # where M is the number of training images
    # and h and w are the height and width of a single image
    training_image_matrix = np.hstack(training_image_vectors)

    # The average face found by taking the mean of all training images
    # (not to be confused with a mean face, i.e. someone frowning)
    # This is denoted by Psi in the Turk and Pentland paper
    mean_face = np.mean(training_image_matrix, axis=1)[:, np.newaxis]
    output_image(mean_face, height, width, output_dir, "mean_face.jpg")

    # Subtract the mean face from each training image
    A = training_image_matrix - mean_face

    # Finding the eigenvectors of the covariance matrix C = A @ A.T is prohibitively
    # expensive, so instead we use this trick to find the first M eigenvectors
    # L has dimension M x M
    # V is the M x M matrix whose columns are the eigenvectors of L
    # U is the (h*w) x M matrix whose columns are the first M eigenvectors of C,
    # i.e. the eigenfaces
    L = A.T @ A
    w, V = np.linalg.eig(L)
    U = A @ V

    for i in range(M):
        output_image(U[:, i], height, width, output_dir, f"eigenface{i}.jpg")

    # The columns of this M x M matrix are the projections of the training images
    # onto the face space, ie the values are the Eigenface coefficients 
    training_coeffs = np.array([U.T @ A[:, i] for i in range(M)]).T
    print("Eigenface coefficients of the training images") #todo formatting
    print(training_coeffs)


    # Recognition
    # test_images = load_images(test_dir)
    # test_image_vectors = images_to_col_vectors(test_images) # todo see above
    test_files = os.listdir(test_dir)
    test_images = []
    for filename in test_files:
        img_path = os.path.join(test_dir, filename)
        img = Image.open(img_path)
        test_images.append(img)

    test_image_vectors = [] # todo remove this
    for img in test_images:
        column_vector = np.array(img.getdata())[:, np.newaxis]
        test_image_vectors.append(column_vector)

    for i, test_img in enumerate(test_image_vectors):
        mean_adjusted_img = test_img - mean_face
        projection = U.T @ mean_adjusted_img #todo output these coefficients
        differences = training_coeffs - projection
        distances = np.linalg.norm(differences, axis=0)
        classification = np.argmin(distances)
        print(f"Classify test image {i} {test_files[i]} as training image {classification} {training_files[classification]}")

        reconstructed = U @ projection
        output_image(reconstructed, height, width, output_dir, f"reconstructed_{i}.jpg")

if __name__ == "__main__":
    main()
