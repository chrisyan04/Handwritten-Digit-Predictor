# Handwritten-Digit-Predictor
This is a small project designed and developed to predict handwritten numbers read from a PNG file using machine and deep learning with Python. The project showcases uses of TensorFlow, Keras, Matplotlib, and Numpy Python libraries to train the model for the predictor.

## Development Summary:
The program implements a neural network model using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset. It follows the following steps:
1. Libraries are imported, including `os`, `cv2`, `numpy`, `matplotlib.pyplot`, and `tensorflow`.
2. The MNIST dataset is loaded, and the data is divided into training and testing sets.
3. The pixel values of the images in the dataset are normalized between 0 and 1.
4. A neural network model is created using the `tf.keras.models.Sequential()` class.
5. Layers are added to the model, including a `Flatten` layer to convert 2D images into a 1D array, two `Dense` layers with ReLU activation, and a final `Dense` layer with softmax activation.
6. The model is compiled with an optimizer, loss function, and evaluation metrics.
7. The model is trained on the training data for a specified number of epochs.
8. The trained model is saved to a file and then loaded back from the file.
9. The model's performance is evaluated using the testing data, and the loss and accuracy values are printed.
10. Handwritten digit images are processed one by one in a loop.
11. Each image is read using OpenCV's `cv2.imread()` function and converted to grayscale.
12. The image is preprocessed by inverting and reshaping it.
13. The trained model predicts the probabilities of each digit class for the image.
14. The predicted digit and the image are displayed using Matplotlib's `imshow()` and `show()` functions.
15. Error handling is implemented to catch any exceptions that may occur during image processing or prediction.
16. The loop continues until there are no more images found in the expected directory.

## Report Summary:
The program provides a basic implementation for trainging and utilizing a neural network model to recognize handwritten digits. The training and testing of the model resulted in precisely predicting any given number in PNG format with an accuracy rate of `92%`, a `16% improvement` from a simple online cam scanner with a rough `76%` accuracy rate.
