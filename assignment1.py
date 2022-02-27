from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2 as cv
import os


# # Read Predictive Image
image_path=input('Enter Image with file name ')

img=cv.imread(image_path)

# # Model Prediction
result=DeepFace.analyze(img, actions = ["age", "emotion", "race","gender"])

# # Visualization
plt.imshow(img[:,:,::-1]) #BRG to RGB
plt.title("2k19-BSCS-202\n" "Hello "+result['gender'])

# # Image Analyzer

print("Age : ",result['age'])
print("Race : ",result['dominant_race'])
print("Emotion : ",result['dominant_emotion'])
plt.show()