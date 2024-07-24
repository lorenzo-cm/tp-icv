from skimage.feature import hog
import numpy as np

from utils.char_extraction import extract_characters


def compute_hog_X(array_characters):
    return np.array([compute_hog_character(char) for char in array_characters])
        

def compute_hog_character(image_character):
    """
    It receives a character image and returns the HOG descriptor for it
    """
    fd = hog(image_character, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), visualize=False)
    
    return np.array(fd)


def compute_hog_full_image(image):
    """
    It receives an image and returns the HOG descriptor 
    for each character in the image (CAPTCHA)
    
    Uses extract_characters with a good custom config
    """
    character_images = extract_characters(image,
                                          width=25,
                                          overlap=2,
                                          space_between=5,
                                          initial_space=10)
    hog_features = []
    for img in character_images:
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)