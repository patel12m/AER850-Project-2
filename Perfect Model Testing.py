import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the previously trained and saved model
model = load_model('C:/Users/patel/Documents/GitHub/AER850-Project-2/Project 2 Data/Perfectmodel.h5')

# Function to process and predict on a single image
def process_and_predict_image(img_path, model):
    # Load the image
    img = image.load_img(img_path, target_size=(100, 100))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Add a batch dimension and rescale
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    # Make predictions
    predictions = model.predict(img_array)
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    return predictions, predicted_class

# Function to display the prediction
def display_prediction(img_path, model, class_labels):
    predictions, predicted_class = process_and_predict_image(img_path, model)
    # Load the image for displaying
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off') 

    # Show the prediction probabilities for each class
    plt.title(f"Predicted: {class_labels[predicted_class[0]]}\n" +
              "\n".join([f"{class_labels[i]}: {predictions[0][i]*100:.2f}%" for i in range(len(class_labels))]))
    plt.show()

# Define the path to your test image and class labels
test_image_path1 = 'C:/Users/patel/Documents/GitHub/AER850-Project-2/Project 2 Data/Data/Test/Medium/Crack__20180419_06_19_09,915.bmp'
class_labels1 = ['Large Crack', 'Medium Crack', 'Small Crack', 'No Crack']  

test_image_path2 = 'C:/Users/patel/Documents/GitHub/AER850-Project-2/Project 2 Data/Data/Test/Large/Crack__20180419_13_29_14,846.bmp'
class_labels2 = ['Large Crack', 'Medium Crack', 'Small Crack', 'No Crack']  

# Display the model's prediction for the test image
display_prediction(test_image_path1, model, class_labels1)