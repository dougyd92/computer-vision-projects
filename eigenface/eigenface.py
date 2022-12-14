# Doug de Jesus
# drd8913@nyu.edu
# N14928011

import argparse
import numpy as np
import os

from PIL import Image


def output_image(gray_values, height, width, directory, filename):
    if directory is not None:
        pixels = np.uint8(np.reshape(gray_values, (height, width)))
        img = Image.fromarray(pixels, mode='L')
        img_path = os.path.join(directory, filename)
        img.save(img_path)


def main():
    parser = argparse.ArgumentParser(
        prog = 'eigenface.py',
        description = """Facial recognition using eigenfaces. Specify a set of training
                         set images and another set of test images to be classified.""")
    parser.add_argument('training_dir', help='Directory containing images for the training set.')
    parser.add_argument('test_dir', help='Directory containing images for the test set.')
    parser.add_argument('-o', '--output_dir', required=False, 
        help='Directory where mean face and eigenfaces will be stored.')
    args = parser.parse_args()

    training_dir = args.training_dir
    test_dir = args.test_dir
    output_dir = args.output_dir

    # Load training images
    training_files = os.listdir(training_dir)
    training_images = []
    for filename in training_files:
        img_path = os.path.join(training_dir, filename)
        img = Image.open(img_path)
        training_images.append(img)

    M = len(training_images)
    height = training_images[0].height
    width = training_images[0].width

    # Get training images in column vector form
    training_image_vectors = []
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
    eigenpairs = {w[i] : V[:, i] for i in range(M)}
    # Sort the eigenvectors by descending eigenvalue
    V = np.vstack([eigenpairs[eigenval] for eigenval in sorted(w, reverse=True)]).T
    U = A @ V

    for i in range(M):
        output_image(U[:, i], height, width, output_dir, f"eigenface{i}.jpg")

    # The columns of this M x M matrix are the projections of the training images
    # onto the face space, ie the values are the Eigenface coefficients 
    training_coeffs = np.array([U.T @ A[:, i] for i in range(M)]).T
    for i in range(M):
        print(f"Training image {training_files[i]} coefficients")
        print(training_coeffs[:, i].reshape(-1,1), "\n")

    # Load test images
    test_files = os.listdir(test_dir)
    test_image_vectors = []
    for filename in test_files:
        img_path = os.path.join(test_dir, filename)
        img = Image.open(img_path)
        column_vector = np.array(img.getdata())[:, np.newaxis]
        test_image_vectors.append(column_vector)

    # Classify test images
    for i, test_img in enumerate(test_image_vectors):
        mean_adjusted_img = test_img - mean_face

        # Project the test image onto the face space to find the eigenface coefficients
        projection = U.T @ mean_adjusted_img #todo output these coefficients

        # Find nearest neighbor in the training set
        differences = training_coeffs - projection
        distances = np.linalg.norm(differences, axis=0)
        classification = np.argmin(distances)

        print(f"Test image {test_files[i]} coefficients")
        print(projection)
        print(f"Closest match is training image {training_files[classification]}\n")

        reconstructed = U @ projection
        output_image(reconstructed, height, width, output_dir, f"reconstructed_{i}.jpg")

if __name__ == "__main__":
    main()
