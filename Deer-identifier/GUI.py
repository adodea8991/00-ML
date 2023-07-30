import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Set image dimensions
img_width, img_height = 224, 224

# Load the trained model
model = load_model('/Users/macbookair/Desktop/Combined-data-set/deer_identifier_model.h5')

# Function to perform prediction
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((img_width, img_height))
        image = ImageTk.PhotoImage(image)
        image_label.configure(image=image)
        image_label.image = image

        # Preprocess the image and make the prediction
        image = Image.open(file_path)
        image = image.resize((img_width, img_height))
        image = img_to_array(image)
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)

        prediction = model.predict(image)[0][0]
        if prediction > 0.5:
            result_label.configure(text="Prediction: Deer (Confidence: {:.2f}%)".format(prediction * 100))
        else:
            result_label.configure(text="Prediction: Not Deer (Confidence: {:.2f}%)".format((1 - prediction) * 100))

# Create the GUI window
window = tk.Tk()
window.title("Deer Identifier")

# Create an image label to display the selected image
image_label = tk.Label(window)
image_label.pack()

# Create a button to select the image
select_button = tk.Button(window, text="Select Image", command=predict_image)
select_button.pack()

# Create a label to display the prediction result
result_label = tk.Label(window, text="")
result_label.pack()

window.mainloop()
