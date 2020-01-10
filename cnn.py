#importting the Keras Libraries and packages
from keras .models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()
#Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Addition of Second convolutional layers and pooling to get more accuracy on the test_set
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1 ,activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Image preprocessing-fitting the CNN to image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('F:\My Machine Learning files\Workshop\DatasetEye\Train',
                                                   target_size=(64, 64),
                                                    batch_size=9,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('F:\My Machine Learning files\Workshop\DatasetEye\Test',
                                                target_size=(64, 64),
                                                batch_size=9,
                                                class_mode='binary')

history =classifier.fit_generator(training_set,
                            steps_per_epoch=575,
                            epochs=4,
                            validation_data=test_set,
                            nb_val_samples=119)
#classification predict
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('F:\My Machine Learning files\Workshop\DatasetEye/Test/Cataract/imgk4.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'normal'
else:
    prediction = 'cataract'
    
print(prediction)

# classifier.save('Cataract_and_Normal.h5')


#matplot

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
