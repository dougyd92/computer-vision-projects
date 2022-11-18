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

def segment_image(img, thresholds):
    gray_values = [grayscale(rgb) for rgb in img.getdata()]

    pixels = np.zeros((len(gray_values), 3))
    print(thresholds, len(thresholds))
    for i in range(len(gray_values)):
        if gray_values[i] <= thresholds[0]:
            # background pixel, display as gray
            pixels[i] = np.full((1,3),gray_values[i])
        elif len(thresholds) < 2 or gray_values[i] <= thresholds[1]:
            # foreground region A, display as red
            pixels[i] = np.array([gray_values[i], 0, 0])
        elif len(thresholds) < 3 or gray_values[i] <= thresholds[2]:
            # foreground region B, display as blue
            pixels[i] = np.array([0, 0, gray_values[i]])            
        else:
            # foreground region C, display as green
            pixels[i] = np.array([0, gray_values[i], 0])

    pixels = np.uint8(np.reshape(pixels, (img.height, img.width, 3)))
    img2 = Image.fromarray(pixels, mode='RGB')

    return img2

def main():
    img = Image.open('../test_images/tiger1.bmp')
    gray_values = [grayscale(rgb) for rgb in img.getdata()]
    histogram = normalized_histogram(gray_values)

    two_region_min_variance = np.inf
    two_region_best_thresholds = []
    three_region_min_variance = np.inf
    three_region_best_thresholds = []
    four_region_min_variance = np.inf
    four_region_best_thresholds = []

    for t1 in range(0, 255):
        classA_var = class_variance(histogram, 0, t1)
        classA_prob = class_probability(histogram, 0, t1)
        classB_var = class_variance(histogram, t1+1, 255)
        classB_prob = class_probability(histogram, t1+1, 255)
        total_var = classA_var * classA_prob + classB_var * classB_prob
        if total_var < two_region_min_variance:
            two_region_min_variance = total_var
            two_region_best_thresholds = [t1]

        for t2 in range(t1+1, 255):
            classB_var = class_variance(histogram, t1+1, t2)
            classB_prob = class_probability(histogram, t1+1, t2)
            classC_var = class_variance(histogram, t2+1, 255)
            classC_prob = class_probability(histogram, t2+1, 255)
            total_var = classA_var * classA_prob + classB_var * classB_prob + classC_var * classC_prob
            if total_var < three_region_min_variance:
                three_region_min_variance = total_var
                three_region_best_thresholds = [t1, t2]
    
            for t3 in range(t2+1, 255):
                classC_var = class_variance(histogram, t2+1, t3)
                classC_prob = class_probability(histogram, t2+1, t3)
                classD_var = class_variance(histogram, t3+1, 255)
                classD_prob = class_probability(histogram, t3+1, 255)
                total_var = classA_var * classA_prob + classB_var * classB_prob + classC_var * classC_prob + classD_var * classD_prob
                if total_var < four_region_min_variance:
                    four_region_min_variance = total_var
                    four_region_best_thresholds = [t1, t2, t3]
    
    print(f"Two regions: {two_region_min_variance}, {two_region_best_thresholds}")
    print(f"Three regions: {three_region_min_variance}, {three_region_best_thresholds}")
    print(f"Four regions: {four_region_min_variance}, {four_region_best_thresholds}")

    if two_region_min_variance <= three_region_min_variance and two_region_min_variance <= four_region_min_variance:
        print(f"Two regions is best, with total weighted variance {two_region_min_variance} using thresholds {two_region_best_thresholds}")
    elif three_region_min_variance <= four_region_min_variance:
        print(f"Three regions is best, with total weighted variance {two_region_min_variance} using thresholds {three_region_best_thresholds}")
    else:
        print(f"Four regions is best, with total weighted variance {four_region_min_variance} using thresholds {four_region_best_thresholds}")

    segmented = segment_image(img, four_region_best_thresholds)
    segmented.show()

if __name__ == "__main__":
    main()
