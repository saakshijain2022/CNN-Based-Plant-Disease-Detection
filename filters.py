import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('training\PlantVillage\demo.JPG')
# Load the image (Replace 'path_to_image.jpg' with the actual path to your image)
I = cv2.imread('training\PlantVillage\demo.JPG', cv2.IMREAD_GRAYSCALE)

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian_noise)
    return noisy_image

# Function to add salt and pepper noise
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size
 
    # Add Salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 255

    # Add Pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0

    return noisy_image

# Add Gaussian noise to the image
J = add_gaussian_noise(I)
plt.subplot(3, 3, 2)
plt.imshow(J, cmap='gray')
plt.title('Gaussian Noised Image')

# Add Salt & Pepper noise to the image
K = add_salt_and_pepper_noise(I, salt_prob=0.4, pepper_prob=0.4)
plt.subplot(3, 3, 3)
plt.imshow(K, cmap='gray')
plt.title('Salt and Pepper Noised Image')

# Function to apply average filter
def apply_average_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

# Apply average filter to Gaussian noisy image
A = apply_average_filter(J, 3)
plt.subplot(3, 3, 4)
plt.imshow(A, cmap='gray')
plt.title('Average Filter on Gaussian Noised Image')

# Apply 5x5 average filter to salt and pepper noisy image
B = apply_average_filter(K, 5)
plt.subplot(3, 3, 5)
plt.imshow(B, cmap='gray')
plt.title('Average Filter of 5x5 on Salt & Pepper Noised Image')

# Apply 7x7 average filter to Gaussian noisy image
C = apply_average_filter(J, 7)
plt.subplot(3, 3, 6)
plt.imshow(C, cmap='gray')
plt.title('Average Filter of 7x7 on Gaussian Noised Image')

# Apply 7x7 average filter to salt and pepper noisy image
D = apply_average_filter(K, 7)
plt.subplot(3, 3, 7)
plt.imshow(D, cmap='gray')
plt.title('Average Filter of 7x7 on Salt & Pepper Noised Image')

# Apply 11x11 average filter to salt and pepper noisy image
E = apply_average_filter(K, 11)
plt.subplot(3, 3, 8)
plt.imshow(E, cmap='gray')
plt.title('Average Filter of 11x11 on Salt & Pepper Noised Image')

# Apply 11x11 average filter to Gaussian noisy image
F = apply_average_filter(J, 11)
plt.subplot(3, 3, 9)
plt.imshow(F, cmap='gray')
plt.title('Average Filter of 11x11 on Gaussian Noised Image')

# Show the plots
plt.tight_layout()
plt.show()
