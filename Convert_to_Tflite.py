##For window user ## Colab .h5 conversion ####
##Run this code in google colab notebook##
##you can able to load file to colab##
from google.colab import files
uploaded = files.upload()

##This converts .h5 file to .tflite file and downloads automatically###
from tensorflow.contrib import lite
from google.colab import files
converter = lite.TFLiteConverter.from_keras_model_file( 'C_and_N.h5' ) 
model = converter.convert()
file = open( 'model.tflite' , 'wb' ) 
file.write( model )
files.download('model.tflite')