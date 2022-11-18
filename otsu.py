# Doug de Jesus
# drd8913@nyu.edu
# N14928011

from PIL import Image
import numpy as np

def grayscale(r, g, b):
    return round(0.299 * r + 0.587 * g + 0.114 * b)

def normalized_histogram(gray_values):
    histogram = {}
    for i in range(256):
        histogram[i] = 0

    for g in gray_values:
        histogram[g] += 1

    # Normalize
    for i in range(256):
        histogram[i] /= len(gray_values)
    
    return histogram

def main():
    img = Image.open('../test_images/tiger1.bmp')

    gray_values = [grayscale(rgb[0], rgb[1], rgb[2]) for rgb in img.getdata()]

    histogram = normalized_histogram(gray_values)

    # gray_values = np.uint8(np.reshape(gray_values, (img.height, img.width)))
    # img2 = Image.fromarray(gray_values, mode='L')
    # img2.show()

    for t1 in range(0, 255):
        print(f"2 regions, [0, {t1}] [{t1+1}, 255]")
        for t2 in range(t1+1, 255):
            print(f"3 regions,  [0, {t1}] [{t1+1}, {t2}] [{t2+1}, 255]")
            for t3 in range(t2+1, 255):
                print(f"4 regions,  [0, {t1}] [{t1+1}, {t2}] [{t2+1}, {t3}] [{t3+1}, 255]")

if __name__ == "__main__":
    main()