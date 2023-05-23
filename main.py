import cv2
import glob
import os
import numpy as np
from PIL import Image

current_dir = os.getcwd()
image_files = [file for file in os.listdir(current_dir) if file.lower().endswith('.jpg')]

# Counter for image numbering
img_number = 1
img_number2 = 1

for file in image_files:
    image_path = os.path.join(current_dir, file)
    output_path = os.path.join(current_dir, f'output{img_number}.jpg')
    output_path2 = os.path.join(current_dir, f'output2{img_number2}.jpg')

    original_image = cv2.imread(file)

    # Grayscale image for the contours
    grayscaled_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Calculate sharpness
    gy, gx = np.gradient(grayscaled_image.astype(np.float32))
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = np.average(gnorm)
    print("Sharpness:", sharpness)

    # Calculate brightness
    rgb_image = Image.open(file).convert('RGB')
    pixel_rgb = rgb_image.getpixel((0, 0))
    brightness = sum(pixel_rgb) / 3
    print("Brightness:", brightness)

    # Gaussian blur threshold
    gaussian_blur_thr = 13 if sharpness <= 14 else 23

    # Blur image
    blurred_image = cv2.GaussianBlur(grayscaled_image, (gaussian_blur_thr, gaussian_blur_thr), 0)

    # Calculate Canny thresholds
    sigma = 0.7
    median = np.median(blurred_image)
    lower = int(max(0, ((1.0 - sigma) * median)))
    upper = int(min(255, ((1.0 + sigma) * median)) - 175)
    print("lower is ", lower)
    print("upper is ", upper)

    # Apply Canny edge detection
    canny_image = cv2.Canny(blurred_image, lower, upper)

    # Threshold the image to black and white
    _, black_white_image = cv2.threshold(canny_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(black_white_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw contours on the original image
    final_image = cv2.drawContours(original_image, contours, -1, (0, 255, 0), 3)

    # Save the black and white image
    cv2.imwrite(output_path2, black_white_image)
    img_number2 += 1

    # Save the contour image
    cv2.imwrite(output_path, final_image)
    img_number += 1

cv2.destroyAllWindows()
