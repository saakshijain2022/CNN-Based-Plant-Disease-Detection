import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and display the original image
img = cv2.imread('./PlantVillage/Tomato_healthy/017a4026-813a-4983-887a-4052bb78c397___RS_HL 0218.JPG')

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding visualization
LT, UT = 100, 150
I_output = np.where((gray_img > LT) & (gray_img < UT), 255, 0).astype(np.uint8)

# Plot thresholding visualization
plt.plot(np.arange(256), np.where((np.arange(256) > LT) & (np.arange(256) < UT), 255, 0))
plt.title('Thresholding Visualization')
plt.xlabel('Pixel Intensity')
plt.ylabel('Output Value')
plt.grid()
plt.show()

# Display original and grayscale images
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Color Image')
plt.subplot(122), plt.imshow(gray_img, cmap='gray'), plt.title('Grayscale Image')
plt.show()

# Spatial Resolution
small_img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
large_img = cv2.resize(small_img, (500, 500), interpolation=cv2.INTER_LINEAR)

plt.subplot(121), plt.imshow(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)), plt.title('Low Res Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB)), plt.title('Upscaled Image')
plt.show()

# Image Enhancement: Digital Negative
negative_img = 255 - gray_img
plt.imshow(negative_img, cmap='gray')
plt.title('Digital Negative')
plt.show()

# Log and Power Law Transformations
c = 255 / np.log(1 + np.max(gray_img))
log_transformed = c * (np.log(1 + gray_img))
power_law_transformed = np.array(255 * (gray_img / 255) ** 0.5, dtype='uint8')

plt.subplot(121), plt.imshow(log_transformed, cmap='gray'), plt.title('Log Transformation')
plt.subplot(122), plt.imshow(power_law_transformed, cmap='gray'), plt.title('Power Law Transformation')
plt.show()

# Smoothing and Sharpening
smoothed_img = cv2.GaussianBlur(img, (5, 5), 0)
sharpened_img = cv2.Laplacian(gray_img, cv2.CV_64F)

plt.subplot(121), plt.imshow(cv2.cvtColor(smoothed_img, cv2.COLOR_BGR2RGB)), plt.title('Smoothed Image')
plt.subplot(122), plt.imshow(sharpened_img, cmap='gray'), plt.title('Sharpened Image')
plt.show()

# Histogram Equalization
equalized_img = cv2.equalizeHist(gray_img)
plt.subplot(121), plt.imshow(gray_img, cmap='gray'), plt.title('Original Grayscale Image')
plt.subplot(122), plt.imshow(equalized_img, cmap='gray'), plt.title('Histogram Equalized Image')
plt.show()

# Discrete Cosine Transform (DCT)
dct = cv2.dct(np.float32(gray_img))
inverse_dct = cv2.idct(dct)

plt.subplot(121), plt.imshow(dct, cmap='gray'), plt.title('DCT')
plt.subplot(122), plt.imshow(inverse_dct, cmap='gray'), plt.title('Inverse DCT')
plt.show()

# Morphological Processing
kernel = np.ones((5, 5), np.uint8)
dilated_img = cv2.dilate(gray_img, kernel, iterations=1)
eroded_img = cv2.erode(gray_img, kernel, iterations=1)


# 5. Edge Detection
# Using Sobel and Prewitt Operators:
# Sobel Edge Detection
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5) # Horizontal edges
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5) # Vertical edges

plt.subplot(121), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(122), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.show()

# 6. Image Segmentation
# Thresholding and Otsu's Method:
# Simple thresholding
ret,thresh1 = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,thresh2 = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.subplot(121), plt.imshow(thresh1, cmap='gray'), plt.title('Simple Thresholding')
plt.subplot(122), plt.imshow(thresh2, cmap='gray'), plt.title('Otsu Thresholding')
plt.show()


# opening_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
# closing_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)

# plt.subplot(121), plt.imshow(dilated_img, cmap='gray'), plt.title('Dilated Image')
# plt.subplot(122), plt.imshow(eroded_img, cmap='gray'), plt.title('Eroded Image')
# plt.show()

# # Edge Detection
# sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
# sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges

# plt.subplot(121), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
# plt.subplot(122), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
# plt.show()

# # Image Segmentation
# ret, thresh1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
# ret2, thresh2 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plt.subplot(121), plt.imshow(thresh1, cmap='gray'), plt.title('Simple Thresholding')
# plt.subplot(122), plt.imshow(thresh2, cmap='gray'), plt.title('Otsu Thresholding')
# plt.show()
