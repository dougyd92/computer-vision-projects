# Doug de Jesus
# drd8913@nyu.edu
# N14928011

from PIL import Image
import numpy as np

def grayscale(rgb):
    """
    Converts a given tuple of (r,g,b) values into a single
    grayscale intensity value using a weighted sum.
    """
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return round(0.299 * r + 0.587 * g + 0.114 * b)

def normalized_histogram(gray_values):
    """
    Given a 1D list of gray values, returns a normalized histogram,
    ie the proportion of pixels in the image corresponding to each
    possible gray level value.
    """
    histogram = {}
    for i in range(256):
        histogram[i] = 0

    for g in gray_values:
        histogram[g] += 1

    # Normalize
    for i in range(256):
        histogram[i] /= len(gray_values)
    
    return histogram

def class_probability(histogram, g_min, g_max):
    """
    For a class that includes all the pixels with gray level values between
    g_min and g_max inclusive, returns the probability that any pixel in 
    the image is in that class.
    """
    sum = 0
    for i in range(g_min, g_max+1):
        sum += histogram[i]
    return sum

def class_mean(histogram, g_min, g_max):
    """
    For a class that includes all the pixels with gray level values between
    g_min and g_max inclusive, returns the mean gray level value for that
    class.
    """
    sum = 0
    for i in range(g_min, g_max+1):
        sum += i * histogram[i]
    return sum / class_probability(histogram, g_min, g_max)

def class_variance(histogram, g_min, g_max):
    """
    For a class that includes all the pixels with gray level values between
    g_min and g_max inclusive, returns the variance of that class.
    """
    sum = 0
    mean = class_mean(histogram, g_min, g_max)
    for i in range(g_min, g_max+1):
        sum += histogram[i] * (i - mean)**2
    return sum / class_probability(histogram, g_min, g_max)

def main():
    img = Image.open('../test_images/tiger1.bmp')

    gray_values = [grayscale(rgb) for rgb in img.getdata()]

    histogram = normalized_histogram(gray_values)

    # gray_values = np.uint8(np.reshape(gray_values, (img.height, img.width)))
    # img2 = Image.fromarray(gray_values, mode='L')
    # img2.show()

    for t1 in range(0, 255):
        classA_var = class_variance(histogram, 0, t1)
        classA_prob = class_probability(histogram, 0, t1)
        classB_var = class_variance(histogram, t1+1, 255)
        classB_prob = class_probability(histogram, t1+1, 255)
        total_var = classA_var * classA_prob + classB_var * classB_prob

        print(f"Total:{total_var} | A:{classA_var:4.4f}, B:{classB_var:4.4f} 2 regions, [0, {t1}] [{t1+1}, 255]")
        # for t2 in range(t1+1, 255):
            # classA_mean = class_mean(histogram, 0, t1)
            # classB_mean = class_mean(histogram, t1+1, t2)
            # classC_mean = class_mean(histogram, t2+1, 255)
            # print(f"[0, {t1}]:{classA_mean:4.4f}, [{t1+1}, {t2}]:{classB_mean:4.4f}, [{t2+1}, 255]:{classC_mean:4.4f}")
        #     # classA_prob = class_probability(histogram, 0, t1)
        #     classB_prob = class_probability(histogram, t1+1, t2)
        #     # classC_prob = class_probability(histogram, t2+1, 255)
        #     # print(f"{round(classA_prob+classB_prob+classC_prob, 2)}={classA_prob}+{classB_prob}+{classC_prob} 3 regions,  [0, {t1}] [{t1+1}, {t2}] [{t2+1}, 255]")
        #     for t3 in range(t2+1, 255):
        #         # classA_prob = class_probability(histogram, 0, t1)
        #         # classB_prob = class_probability(histogram, t1+1, t2)
        #         classC_prob = class_probability(histogram, t2+1, t3)
        #         classD_prob = class_probability(histogram, t3+1, 255)
        #         print(f"{round(classA_prob+classB_prob+classC_prob+classD_prob, 4)}={classA_prob}+{classB_prob}+{classC_prob}+{classD_prob} 4 regions,  [0, {t1}] [{t1+1}, {t2}] [{t2+1}, {t3}] [{t3+1}, 255]")

if __name__ == "__main__":
    main()