# Author @ Nydia R. Varela-Rosales, M. Engel
# Version v1 2024
# Description : predicts cqc probability score based on FFT from simulation snapshots at different epsilon and T values
# Requires    : tensorflow, keras, numpy, data FFT images

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Parameters
inputModel = 'best_chirality_detector_early.keras' # Input name of the model
sizeImage  = 512                                   # Image size
scaleInput = 'grayscale'                           # Color scale

def predictChirality(img_path):
    model      = load_model(inputModel)  # load model
    img        = image.load_img(img_path, target_size=(sizeImage, sizeImage), color_mode=scaleInput) # Load image
    imgArray   = image.img_to_array(img) # convert to array
    imgArray   = np.expand_dims(imgArray, axis=0) / 255.0
    prediction = model.predict(imgArray) # predict
    return prediction[0][0] # output label


img_path            = sys.argv[1]                # Path to image director
cqcProbabilityScore = predictChirality(img_path) # Predict cqc
newFile             = open(img_path.replace(".png","_m2.dat"),"w") # set-up output file
newFile.write(str(cqcProbabilityScore)) # Write prediction to a file FILENAME+"_m2.dat"
