import cv2
import numpy as np

def preprocess_image(img_path=None, image_bytes=None):
    """
    Preprocess image following the thesis pipeline.
    Reads image, applies Gaussian Blur, converts to HSV, creates a green color mask,
    applies morphological closing, and segments the background.
    """
    if img_path:
        img = cv2.imread(img_path)
    elif image_bytes:
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Either img_path or image_bytes must be provided.")

    if img is None:
        return None

    # Resume the 224x224 shape to match the training models
    img_resized = cv2.resize(img, (224, 224))
    
    # Gaussian blur
    blurImg = cv2.GaussianBlur(img_resized, (5, 5), 0)   
    
    # Convert to HSV image
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    
    # Create mask (parameters - green color range)
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Create bool mask
    bMask = mask > 0  
    
    # Apply the mask
    clear = np.zeros_like(img_resized, np.uint8)  # Create empty image
    clear[bMask] = img_resized[bMask]  # Apply boolean mask to the origin image
    
    return clear
