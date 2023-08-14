# SM23-AI-ROS-03
Teachable Machine Model

Trains a computer to recognize your own pictures, sounds, and gestures.

The fast and easy way to build machine learning models for your website, app, and more with no technical or coding skills required.

What is a Teachable Machine?


Teachable Machine is a website tool that allows accessible models for anyone quickly, easily and fast to create machine learning models.

# How to use it?

Teachable Machines is flexible - use data or capture patterns in real time. Respecting the way you work.

You can choose to use all of them on the device without any webcam or microphone data leaving from your computer.

+ Photos

Teach the model to classify photos using files or webcams.

+ Sound

Teach a model for classifying sound by writing short sound samples.

+ Poses

Shows a model for classifying body poses or positions using poses from files or webcam. 

+ The models you make with Teachable Machine are real TensorFlow.js models that work anywhere javascript runs, so they play nice with tools like Glitch, P5.js, Node.js & more.

Plus, export to different formats to use your models elsewhere, like Coral, Arduino & more.

# Using the Model
After the model is trained, you can use it for many different purposes.

This example predicts input image, given 3 input images for class named nature, and a class named beach, then calls a function to give a percentage prediction of each class in the input test image based on the given classes:

+ percentage prediction result:
  + First image prediction result:
    <img width="960" alt="2023-08-10 (3)" src="https://github.com/Naif-Al-Ajlani/SM23-AI-ROS-03/assets/98528261/2cf7acb5-b2c9-4fe2-a531-f888b6133f6a">

  + Seconed image prediction result:
    <img width="956" alt="2023-08-10 (4)" src="https://github.com/Naif-Al-Ajlani/SM23-AI-ROS-03/assets/98528261/b9d34659-eb1c-4602-932d-d5ac6f178fbc">

  + Third image prediction result:
    <img width="960" alt="2023-08-10 (5)" src="https://github.com/Naif-Al-Ajlani/SM23-AI-ROS-03/assets/98528261/a5888e6b-2043-4973-8289-5423c97cfcad">

# Open CV Keras code used to train the model:

+ Note(TensorFlow is required for Keras to work)
 ```
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

```

# Resorces:
+ Machine model training site: https://teachablemachine.withgoogle.com
  
+ Teachable Machine Tutorial: Snap, Clap, Whistle: https://medium.com/@warronbebster/teachable-machine-tutorial-snap-clap-whistle-4212fd7f3555
  
+ Teachable Machine Tutorial: Head Tilt: https://medium.com/@warronbebster/teachable-machine-tutorial-head-tilt-f4f6116f491
