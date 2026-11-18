import cv2
import numpy as np
from skimage import exposure

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Gaussian Filter (noise reduction)
    img_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    
    # CLAHE (contrast enhancement)
    lab = cv2.cvtColor(img_gauss, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced
