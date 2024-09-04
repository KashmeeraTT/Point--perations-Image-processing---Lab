import cv2
import numpy as np
import matplotlib.pyplot as plt

# Replace with your actual index number
index_number = "210279A"

# Load the image
src_image = cv2.imread(f"{index_number}_SrcImage.jpg")
if src_image is None:
    raise FileNotFoundError("Source image not found.")

# Get image size (height, width)


def get_image_size(image):
    return image.shape[:2]


# Convert to grayscale (8-bpp)
def convert_to_grayscale(image):
    height, width = get_image_size(image)
    gray_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            (b, g, r) = image[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_image[i, j] = gray_value
    return gray_image


gray_image = convert_to_grayscale(src_image)


# Negative Image
def create_negative_image(image):
    height, width = get_image_size(image)
    negative_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            negative_image[i, j] = 255 - image[i, j]
    return negative_image


negative_image = create_negative_image(gray_image)


# Increase Brightness by 20%
def increase_brightness(image, factor=1.2):
    height, width = get_image_size(image)
    bright_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            bright_image[i, j] = np.clip(image[i, j] * factor, 0, 255)
    return bright_image


increased_brightness_image = increase_brightness(gray_image)


# Reduce Contrast (Gray levels between 125 and 175)
def reduce_contrast(image, min_gray=10, max_gray=15):
    height, width = image.shape  # assuming image is a grayscale 2D array
    reduced_contrast_image = np.zeros((height, width), dtype=np.uint8)

    # Get min and max once
    img_min = image.min()
    img_max = image.max()

    # Avoid divide by zero in case image has no variation
    if img_max == img_min:
        return np.full_like(image, (min_gray + max_gray) // 2)

    # Normalize and scale each pixel
    for i in range(height):
        for j in range(width):
            normalized_pixel = (image[i, j] - img_min) / (img_max - img_min)
            scaled_pixel = normalized_pixel * (max_gray - min_gray) + min_gray
            reduced_contrast_image[i, j] = np.clip(scaled_pixel, 0, 255)

    return reduced_contrast_image.astype(np.uint8)


reduced_contrast_image = reduce_contrast(gray_image)


# Reduce Gray level depth to 4bpp
def reduce_gray_depth(image):
    height, width = get_image_size(image)
    reduced_gray_depth_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            reduced_gray_depth_image[i, j] = (image[i, j] >> 4) << 4
    return reduced_gray_depth_image


reduced_gray_depth_image = reduce_gray_depth(gray_image)


# Vertical mirror image
def create_vertical_mirror_image(image):
    height, width = get_image_size(image)
    vertical_mirror_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            vertical_mirror_image[i, j] = image[i, width - 1 - j]
    return vertical_mirror_image


vertical_mirror_image = create_vertical_mirror_image(gray_image)

# Save the output images
cv2.imwrite(f"{index_number}_OPImage_1_1.jpg", gray_image)
cv2.imwrite(f"{index_number}_OPImage_1_2.jpg", negative_image)
cv2.imwrite(f"{index_number}_OPImage_1_3.jpg", increased_brightness_image)
cv2.imwrite(f"{index_number}_OPImage_2_1.jpg", reduced_contrast_image)
cv2.imwrite(f"{index_number}_OPImage_2_2.jpg", reduced_gray_depth_image)
cv2.imwrite(f"{index_number}_OPImage_2_3.jpg", vertical_mirror_image)

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Display images in their respective grid positions
axs[0, 0].imshow(gray_image, cmap='gray')
axs[0, 0].set_title('Original Grayscale Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(negative_image, cmap='gray')
axs[0, 1].set_title('Negative Image')
axs[0, 1].axis('off')

axs[0, 2].imshow(increased_brightness_image, cmap='gray')
axs[0, 2].set_title('Increased Brightness')
axs[0, 2].axis('off')

axs[1, 0].imshow(reduced_contrast_image, cmap='gray')
axs[1, 0].set_title('Reduced Contrast')
axs[1, 0].axis('off')

axs[1, 1].imshow(reduced_gray_depth_image, cmap='gray')
axs[1, 1].set_title('Reduced Gray Depth (4bpp)')
axs[1, 1].axis('off')

axs[1, 2].imshow(vertical_mirror_image, cmap='gray')
axs[1, 2].set_title('Vertical Mirror Image')
axs[1, 2].axis('off')

# Save the subplot
plt.savefig(f"{index_number}_SubPlot.jpg")
plt.show()
