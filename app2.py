import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('Allmodel.h5', compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #Compiles the model with loss_fn for better performance with batch processing. 
# ->suitable for multi-class classification.

loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


class_names = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']


BATCH_SIZE = 32
IMAGE_SIZE = 255
CHANNEL = 3
EPOCHS = 50

#  preprocess an input image, and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Converts the input image img  into a NumPy array with pixel values.
    img_array = tf.expand_dims(img_array, 0)
    #  to add an additional dimension to the tensor img_array.
    # before (height, width, channels) now (1, height, width, channels), effectively making it a batch of one image.
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    #  finds the index of the highest probability in the predictions array,
    confidence = round(100 * (np.max(predictions[0])), 2)
    # np.max(predictions[0]) finds the highest probability value in predictions[0] and  rounds the percentage to two decimal places for clarity
    return predicted_class, confidence

# Route to the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index1.html', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index1.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists('static'):
                os.makedirs('static')

            filepath = os.path.join('static', filename)
            file.save(filepath)
            # Checks if a file is uploaded and if it has an allowed filename.
            # Saves the uploaded file to the static folder.

            # Read the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            original_img = cv2.imread(filepath)
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

            ### Thresholding Visualization - setting pixel values within the range [100, 150] to 255 (white) and others to 0 (black).
            LT, UT = 100, 150
            I_output = np.where((gray_img > LT) & (gray_img < UT), 255, 0).astype(np.uint8)

            plt.plot(np.arange(256), np.where((np.arange(256) > LT) & (np.arange(256) < UT), 255, 0))
            plt.title('Thresholding Visualization')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Output Value')
            plt.grid()
            # Saves the threshold visualization plot as an image.
            threshold_viz_path = os.path.join('static', 'threshold_viz.png')
            plt.savefig(threshold_viz_path)
            plt.close()

            ### Original and Grayscale Image
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Color Image')
            ax[1].imshow(gray_img, cmap='gray')
            ax[1].set_title('Grayscale Image')

            img_comparison_path = os.path.join('static', 'img_comparison.png')
            plt.savefig(img_comparison_path)
            plt.close()


            ### Spatial Resolution (Low Res & Upscaled)
            # Spatial Resolution
            # Low-Res and Upscaled Images: Reduces and enlarges the image, demonstrating spatial resolution effects.
            # Digital Negative: Inverts the grayscale image to create a digital negative.
            # Log and Power Law Transformations: Applies logarithmic and power transformations for contrast adjustments.
            # Smoothing and Sharpening: Applies Gaussian smoothing and Laplacian sharpening to demonstrate effects on image clarity.
            # Histogram Equalization: Enhances contrast using histogram equalization.
            # Discrete Cosine Transform (DCT): Applies DCT and inverse DCT for frequency domain analysis.
                

            #  Interpolation Method for Resizing:
            # cv2.INTER_AREA: This is used when we need to shrink an image.
            # cv2.INTER_CUBIC: This is slow but more efficient.
            # cv2.INTER_LINEAR: This is primarily used when zooming is required. This is the default interpolation technique in OpenCV.
            small_img = cv2.resize(original_img, (100, 100), interpolation=cv2.INTER_AREA)
            large_img = cv2.resize(small_img, (500, 500), interpolation=cv2.INTER_LINEAR)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Low Res Image')
            ax[1].imshow(cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB))
            ax[1].set_title('Upscaled Image')

            spatial_resolution_path = os.path.join('static', 'spatial_resolution.png')
            plt.savefig(spatial_resolution_path)
            plt.close()

            ### Digital Negative
            #  L = 256.  s = L-1-r 
            negative_img = 255 - gray_img
            plt.imshow(negative_img, cmap='gray')
            plt.title('Digital Negative')

            digital_negative_path = os.path.join('static', 'digital_negative.png')
            plt.savefig(digital_negative_path)
            plt.close()

            ### Log and Power Law  (Gamma) Transformations
            # c is given by 255/(log (1 + m)), where m is the maximum pixel value in the image. It is done to ensure that the final pixel value does not exceed (L-1), or 255.
            c = 255 / np.log(1 + np.max(gray_img))
            log_transformed = c * (np.log(1 + gray_img))

            # Gamma correction is important for displaying images on a screen correctly, to prevent bleaching or darkening of images when viewed from different types of monitors with different display settings
            power_law_transformed = np.array(255 * (gray_img / 255) ** 0.5, dtype='uint8')
            #  normalizes the pixel values , gamma value of 0.5. ,
            # If gamma < 1, it enhances the darker regions of the image (making them lighter).
            # If gamma > 1, it enhances the lighter regions (making them darker).
            # : After applying the transformation, this line multiplies the resulting values by 255 to scale them back to the original pixel value range of [0, 255].

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(log_transformed, cmap='gray')
            ax[0].set_title('Log Transformation')
            ax[1].imshow(power_law_transformed, cmap='gray')
            ax[1].set_title('Power Law Transformation')

            log_power_law_path = os.path.join('static', 'log_power_law.png')
            plt.savefig(log_power_law_path)
            plt.close()

            ### Smoothing and Sharpening
            smoothed_img = cv2.GaussianBlur(original_img, (5, 5), 0)
            # 0: This is the standard deviation (σ)  , Gaussian kernel = 5,5
            sharpened_img = cv2.Laplacian(gray_img, cv2.CV_64F)
            # cv2.CV_64F: This specifies the data type of the output image. CV_64F indicates that the output will be a 64-bit floating point image. This is important to avoid overflow issues when calculating gradients.  [0,-1,0;-1,4,-1;0,-1,0]
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(cv2.cvtColor(smoothed_img, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Smoothed Image')
            ax[1].imshow(sharpened_img, cmap='gray')
            ax[1].set_title('Sharpened Image')

            smoothing_sharpening_path = os.path.join('static', 'smoothing_sharpening.png')
            plt.savefig(smoothing_sharpening_path)
            plt.close()

            ### Histogram Equalization
            equalized_img = cv2.equalizeHist(gray_img)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(gray_img, cmap='gray')
            ax[0].set_title('Original Grayscale Image')
            ax[1].imshow(equalized_img, cmap='gray')
            ax[1].set_title('Histogram Equalized Image')

            histogram_equalization_path = os.path.join('static', 'histogram_equalization.png')
            plt.savefig(histogram_equalization_path)
            plt.close()

            ### Discrete Cosine Transform (DCT)
            dct = cv2.dct(np.float32(gray_img))
            inverse_dct = cv2.idct(dct)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(dct, cmap='gray')
            ax[0].set_title('DCT')
            ax[1].imshow(inverse_dct, cmap='gray')
            ax[1].set_title('Inverse DCT')

            dct_path = os.path.join('static', 'dct.png')
            plt.savefig(dct_path)
            plt.close()

            ### Prediction using the model  -Uses the predict function to classify the uploaded image and get the confidence score.
            predicted_class, confidence = predict(img)

            # Render the template with all visualizations
            return render_template('index1.html', image_path=filepath,
                                   threshold_viz=threshold_viz_path,
                                   img_comparison=img_comparison_path,
                                   spatial_resolution=spatial_resolution_path,
                                   digital_negative=digital_negative_path,
                                   log_power_law=log_power_law_path,
                                   smoothing_sharpening=smoothing_sharpening_path,
                                   predicted_label=predicted_class,
                                   actual_label=predicted_class,
                                    histogram_equalization=histogram_equalization_path,
                                    dct=dct_path,
                                   confidence=confidence)

    return render_template('index1.html', message='Upload an image')

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
