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

class OtsusSolver:
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

    def get_best_thresholds(self):
        two_region_min_variance = np.inf
        two_region_best_thresholds = []
        three_region_min_variance = np.inf
        three_region_best_thresholds = []
        four_region_min_variance = np.inf
        four_region_best_thresholds = []

        for t1 in range(0, 255):
            classA_var = self.class_variance(0, t1)
            classA_prob = self.class_probability(0, t1)
            classB_var = self.class_variance(t1+1, 255)
            classB_prob = self.class_probability(t1+1, 255)
            total_var = classA_var * classA_prob + classB_var * classB_prob
            if total_var < two_region_min_variance:
                two_region_min_variance = total_var
                two_region_best_thresholds = [t1]

            for t2 in range(t1+1, 255):
                classB_var = self.class_variance(t1+1, t2)
                classB_prob = self.class_probability(t1+1, t2)
                classC_var = self.class_variance(t2+1, 255)
                classC_prob = self.class_probability(t2+1, 255)
                total_var = classA_var * classA_prob + classB_var * classB_prob + classC_var * classC_prob
                if total_var < three_region_min_variance:
                    three_region_min_variance = total_var
                    three_region_best_thresholds = [t1, t2]

                for t3 in range(t2+1, 255):
                    classC_var = self.class_variance(t2+1, t3)
                    classC_prob = self.class_probability(t2+1, t3)
                    classD_var = self.class_variance(t3+1, 255)
                    classD_prob = self.class_probability(t3+1, 255)
                    total_var = classA_var * classA_prob + classB_var * classB_prob + classC_var * classC_prob + classD_var * classD_prob
                    if total_var < four_region_min_variance:
                        four_region_min_variance = total_var
                        four_region_best_thresholds = [t1, t2, t3]

        print(f"Two regions: {two_region_min_variance}, {two_region_best_thresholds}")
        segmented = segment_image(self.img, two_region_best_thresholds)
        segmented.show()
        print(f"Three regions: {three_region_min_variance}, {three_region_best_thresholds}")
        segmented = segment_image(self.img, three_region_best_thresholds)
        segmented.show()
        print(f"Four regions: {four_region_min_variance}, {four_region_best_thresholds}")
        segmented = segment_image(self.img, four_region_best_thresholds)
        segmented.show()

        if two_region_min_variance <= three_region_min_variance and two_region_min_variance <= four_region_min_variance:
            print(f"Two regions is best, with total weighted variance {two_region_min_variance} using thresholds {two_region_best_thresholds}")
        elif three_region_min_variance <= four_region_min_variance:
            print(f"Three regions is best, with total weighted variance {two_region_min_variance} using thresholds {three_region_best_thresholds}")
        else:
            print(f"Four regions is best, with total weighted variance {four_region_min_variance} using thresholds {four_region_best_thresholds}")

        return 0

def main():
    img = Image.open('test_images/tiger1.bmp')

    otsus = OtsusSolver(img)

    otsus.get_best_thresholds()

if __name__ == "__main__":
    main()
