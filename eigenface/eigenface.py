# Doug de Jesus
# drd8913@nyu.edu
# N14928011

import os
import numpy as np
from PIL import Image


def output_image(gray_values, height, width, directory, filename):
    pixels = np.uint8(np.reshape(gray_values, (height, width)))
    img = Image.fromarray(pixels, mode='L')
    img_path = os.path.join(directory, filename)
    img.save(img_path)


def main():
    # todo argparse
    training_dir = "../Project2/Face dataset/Training"
    output_dir = "../Project2/Face dataset/Output"

    training_files = os.listdir(training_dir)

    training_images = []
    for filename in training_files:
        img_path = os.path.join(training_dir, filename)
        img = Image.open(img_path)
        training_images.append(img)

    M = len(training_images)
    height = training_images[0].height
    width = training_images[0].width

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
    U = A @ V

    for i in range(M):
        output_image(U[:, i], height, width, output_dir, f"eigenface{i}.jpg")

    # Eigenface coefficients of the training images
    training_coeffs = [U.T @ A[:, i] for i in range(M)]
    print("Eigenface coefficients of the training images") #todo formatting
    print(training_coeffs)

    # todo recognition
    # use images to produce set of eigenfaces and coefficients
    # can use numpy for computing eigenvals and eigenvectors

    # output:
    #   mean face and eigenfaces
    #   eigenface coeffs of training images

    # recognize faces in test data using 1NN
    # compute euclidean distances between eigenface coeffs of input and of training images
    # argmin
    # can skip I_R, d_00, T_0, T_1 from lecture

    # output:
    #   eigenface coeffs of each test image
    #   recognition result of each test image

if __name__ == "__main__":
    main()
