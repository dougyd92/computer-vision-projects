# Doug de Jesus
# drd8913@nyu.edu
# N14928011

import argparse
import pathlib

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

def remap(value, min, max):
    """
    Remap values in between two thresholds into the full
    [0, 255] range. This helper is just to make the output
    image easier to look at; it does not affect the actual
    segmentation process.
    """
    return 255 * (value - min ) / (max - min)

def get_segmented_image(img, thresholds):
    """
    Given an Image and an array of thresholds,
    returns a new Image where the regions corresponding
    to the different thresholds are displayed as different
    colors.
    Background: Gray
    Foreground region A: Red
    Foreground region B (if applicable): Blue
    Foreground region C (if applicable): Green
    """
    thresholds.append(255)
    
    gray_values = [grayscale(rgb) for rgb in img.getdata()]
    pixels = np.zeros((len(gray_values), 3))

    for i in range(len(gray_values)):
        if gray_values[i] <= thresholds[0]:
            # background pixel, display as grays
            pixels[i] = np.full((1,3),gray_values[i])
        elif gray_values[i] <= thresholds[1]:
            # foreground region A, display as red
            value = remap(gray_values[i], thresholds[0]+1, thresholds[1])
            pixels[i] = np.array([value, 0, 0])
        elif gray_values[i] <= thresholds[2]:
            # foreground region B, display as blue
            value = remap(gray_values[i], thresholds[1]+1, thresholds[2])
            pixels[i] = np.array([0, 0, value])
        else:
            # foreground region C, display as green
            value = remap(gray_values[i], thresholds[2]+1, 255)
            pixels[i] = np.array([0, value, 0])

    pixels = np.uint8(np.reshape(pixels, (img.height, img.width, 3)))
    img2 = Image.fromarray(pixels, mode='RGB')

    return img2

class OtsusSolver:
    """
    A class to implement Otsu's method to find the best thresholds
    to segment a given image into 2, 3, or 4 regions.
    Each instance of this class is initialized with a particular image,
    and memoizes intermediate results for better performance.
    """
    def __init__(self, img):
        self.img = img
        gray_values = [grayscale(rgb) for rgb in img.getdata()]
        self.histogram = normalized_histogram(gray_values)

        # for memoization
        self.class_probs = np.full((256,256), -1, dtype='f')
        self.class_means = np.full((256,256), -1, dtype='f')
        self.class_vars = np.full((256,256), -1, dtype='f')

    def class_probability(self, g_min, g_max):
        """
        For a class that includes all the pixels with gray level values between
        g_min and g_max inclusive, returns the probability that any pixel in
        the image is in that class.
        """
        if self.class_probs[g_min][g_max] < 0: # check if this has been computed already
            sum = 0
            for i in range(g_min, g_max+1):
                sum += self.histogram[i]
            self.class_probs[g_min][g_max] = sum # memoize
        return self.class_probs[g_min][g_max]

    def class_mean(self, g_min, g_max):
        """
        For a class that includes all the pixels with gray level values between
        g_min and g_max inclusive, returns the mean gray level value for that
        class.
        """
        if self.class_means[g_min][g_max] < 0: # check if this has been computed already
            sum = 0
            for i in range(g_min, g_max+1):
                sum += i * self.histogram[i]
            # mean = sum / self.class_probability(g_min, g_max)
            prob = self.class_probability(g_min, g_max)
            mean = sum / prob
            self.class_means[g_min][g_max] = mean # memoize
        return self.class_means[g_min][g_max]

    def class_variance(self, g_min, g_max):
        """
        For a class that includes all the pixels with gray level values between
        g_min and g_max inclusive, returns the variance of that class.
        """
        if self.class_vars[g_min][g_max] < 0: # check if this has been computed already
            sum = 0
            mean = self.class_mean(g_min, g_max)
            for i in range(g_min, g_max+1):
                sum += self.histogram[i] * (i - mean)**2
            variance = sum / self.class_probability(g_min, g_max)
            self.class_vars[g_min][g_max] = variance # memoize
        return self.class_vars[g_min][g_max]

    def get_weighted_total_variance(self, thresholds):
        """
        Compute the sum of weighted variances when segmenting
        the image using the provided thesholds.

        thresholds: an array of 1 or more thresholds
        """
        num_regions = len(thresholds) + 1
        thresholds = [-1] + thresholds + [255] # implicit upper and lower limits

        total_var = 0
        for i in range(num_regions):
            class_var = self.class_variance(thresholds[i] + 1, thresholds[i+1])
            class_prob = self.class_probability(thresholds[i] + 1, thresholds[i+1])
            total_var += class_var * class_prob
        return total_var

    def get_best_thresholds(self):
        """
        Computes the optimal thresholds and total variance
        for segmenting the image into 2, 3, or 4 regions.

        Returns:
        variance_by_num_regions, thresholds_by_num_regions
        variance_by_num_regions (dict: int -> float): sum of weighted variances, keyed by number of regions
        thresholds_by_num_regions (dict: int -> array[int]): ideal thresholds to use, keyed by number of regions
        """
        variance_by_num_regions = {2: np.inf, 3: np.inf, 4: np.inf}
        thresholds_by_num_regions = {2: [], 3: [], 4: []}

        for t1 in range(0, 255):
            total_var = self.get_weighted_total_variance([t1])
            if total_var < variance_by_num_regions[2]:
                variance_by_num_regions[2] = total_var
                thresholds_by_num_regions[2] = [t1]

            for t2 in range(t1+1, 255):
                total_var = self.get_weighted_total_variance([t1, t2])
                if total_var < variance_by_num_regions[3]:
                    variance_by_num_regions[3] = total_var
                    thresholds_by_num_regions[3] = [t1, t2]

                for t3 in range(t2+1, 255):
                    total_var = self.get_weighted_total_variance([t1, t2, t3])
                    if total_var < variance_by_num_regions[4]:
                        variance_by_num_regions[4] = total_var
                        thresholds_by_num_regions[4] = [t1, t2, t3]

        return variance_by_num_regions, thresholds_by_num_regions

def main():
    parser = argparse.ArgumentParser(
                prog = 'otsu.py',
                description = 'Takes a .bmp and outputs another .bmp with the image segmented into 2, 3, or 4 regions')
    parser.add_argument('source_file', type=pathlib.Path, help='The source image to be segmented. Must be a bitmap (.bmp) file')
    parser.add_argument('-d', '--dest', required=False, type=pathlib.Path, help='''The destination file path to save the segmented image. 
        If omitted, the image will be previewed with Image.show()''')
    args = parser.parse_args()

    img = Image.open(args.source_file)
    otsus = OtsusSolver(img)
    variance_by_num_regions, thresholds_by_num_regions = otsus.get_best_thresholds()

    for i in range(2, 5):
        print(f"{i} regions: best threshold(s)={thresholds_by_num_regions[i]}, with total weighted variance {variance_by_num_regions[i]}")

    # Because the total variance will always decrease as the number of regions increases, we apply some tricks
    # to decide whether increasing the number of regions provides enough of a decrease to be sensible. 
    ideal_num_regions = 4
    if thresholds_by_num_regions[3]/thresholds_by_num_regions[2] > 0.5:
        # If using 3 regions does not cut down the variance by at least half
        # compared to using 2 regions, then just stuck with 2
        ideal_num_regions = 2
    elif (thresholds_by_num_regions[3] - thresholds_by_num_regions[4]) / (thresholds_by_num_regions[2] - thresholds_by_num_regions[3]) < 0.33:
        # If the decrease in variance going from 3 to 4 regions is less than a third
        # of the decrease in variance going from 2 to 3 regions, then just stick with 3.
        ideal_num_regions = 3

    segmented_img = get_segmented_image(img, thresholds_by_num_regions[ideal_num_regions])

    if args.dest is not None:
        segmented_img.save(args.dest)
    else:
        segmented_img.show()

if __name__ == "__main__":
    main()
