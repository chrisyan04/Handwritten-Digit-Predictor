import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
# setting up training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize: scale between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# create the neurel network model
model = tf.keras.models.Sequential()

# add some layers to the model
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # Flatten layer: turns into one big line of 28x28 pixels
model.add(tf.keras.layers.Dense(128, activation='relu')) # Dense layer: basic neurel network layer where each neuron is connected another neuron
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # Dense layer: w/ 10 neurons, 'softmax' makes sure that all 10 neurons add up to 1 (like a confidence); prob for each neruon to be right digit {0,1,2,3,4,5,6,7,8,9}

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=10) 

# save the model
model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

# show loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print(loss) # Loss: ~0.1
print(accuracy) # Accuracy: ~0.97
 
image_number = 0
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
