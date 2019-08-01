"""KERAS TUTORIAL - HAPPY HOUSE


--- GOALS ---

1. Learn to use Keras
--> A high-level neural networks API (programming framework)
--> Written in Python and capable of running on top of several lower-level frameworks (including TensorFlow and CNTK)

2. See how you can in a couple of hours build a deep learning algorithm.


--- QUESTIONS ---

1. Why are we using Keras?
--> Keras was developed to enable deep learning engineers to build and experiment with different models very quickly
--> Just as TensorFlow is a higher-level framework than Python, Keras even higher and provides additional abstractions
--> Being able to go from idea to result with the least possible delay is key to finding good models. However, Keras is more restrictive than the lower-level frameworks, so there are some very complex models that you can implement in TensorFlow but not (without more difficulty) in Keras. That being said, Keras will work fine for many common models.


--- TIME TO START ---

In this exercise, you'll work on the "Happy House" problem, which we'll explain below
Let's load the required packages and solve the problem of the Happy House!

--- NOTE ---
--> As you can see, we've imported a lot of functions from Keras
--> This is so you can use them easily just by calling them directly in the notebook
----> Ex: X = Input(...) or X = ZeroPadding2D(...)

"""

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

K.set_image_data_format('channels_last')

'''
The Happy House

For your next vacation, you decided to spend a week with five of your friends from school
It is a very convenient house with many things to do nearby
But the most important benefit is that everybody has committed to be happy when they are in the house
So anyone wanting to enter the house must prove their current state of happiness

As a deep learning expert, to make sure the "Happy" rule is strictly applied
You are going to build an algorithm which that uses pictures from the front door camera to check if the person is happy
The door should open only if the person is happy

You have gathered pictures of your friends and yourself, taken by the front-door camera (See images folder in dir.)
--> y = 0 -- indicates -- 'not happy'
--> y = 1 -- indicates -- 'happy'

Run the following code to normalize the dataset and learn about its shapes

'''

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors (Values between 0-255 become values between 0-1)
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T
print(Y_train)
print("Here's the Y_train before (ROW)\n{}".format(Y_train_orig))
print("Here's the Y_train before (COLUMN)\n{}".format(Y_train))

print("\nNumber of training examples = " + str(X_train.shape[0]))
print("Number of test examples = " + str(X_test.shape[0]))

print("\nX_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))

print("\nX_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# --- Details of the "Happy" Data-Set ---
#
#       Images are of shape (64,64,3)
#       Training: 600 pictures
#       Test: 150 pictures

'''

--- BUILDING A MODEL USING KERAS ---

Keras is very good for rapid prototyping
--> In just a short time you will be able to build a model that achieves outstanding results

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        Here is an example of a model in Keras:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


--- NOTE ---

--> Keras uses a different convention with variable names than we've previously used with numpy and TensorFlow
--> In particular, rather than creating and assigning a new variable on each step of forward propagation...
-----> .. <<< such as X, Z1, A1, Z2, A2, etc. for the computations for the different layers .. >>>
-----> In Keras code each line above just reassigns X to a new value using X = .... 
-----> i.e during each step of forward prop, we are writing the last value in the computation into the same variable X
-----> The only exception was X_input, which we kept separate and did not overwrite, since we needed it at the end
-------> We need X_input to create the Keras model instance (model = Model(inputs = X_input, ...) above)


--- PROGRAM - Implement a HappyModel() ---

This program is more open-ended than most
We suggest that you start by implementing a model using the architecture we suggest
You then can run through the rest of this assignment using that as your initial model

After that, come back and take initiative to try out other model architectures

i.e. You might take inspiration from the model above, then vary the network architecture and hyperparameters as you wish
You can also use other functions such as AveragePooling2D(), GlobalMaxPooling2D(), Dropout()


--- NOTE ---

--> You have to be careful with your data's shapes
--> Use what you've learned to make sure your conv, pool, & fc layers are adapted to the volumes you're applying it to

'''


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.

    return model


'''
You have now built a function to describe your model. To train and test this model, there are four steps in Keras:

1. Create the model by calling the function above
2. Compile the model by calling model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])
3. Train the model on train data by calling model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)
4. Test the model on test data by calling model.evaluate(x = ..., y = ...)

If you want to know more about model.compile(), model.fit(), model.evaluate() and their arguments...
--> refer to the official Keras documentation - online -- or ya know google it

'''

# STEP 1 - CREATE THE MODEL
happyModel = HappyModel(X_train.shape[1:])

# STEP 2 - COMPILE THE MODEL
# --Note-- Adam is a robust optimizer often used for binary classification and binary_crossentropy is the best choice
happyModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# STEP 3 - TRAIN THE MODEL
# --Note-- Batch Size and Epoch Count are Best Guesses Currently
# ----------> Batch Size Being Smaller Means Jittery Gradient Descent --- Larger Means Smoother (but more memory)
# ----------> Epoch counts should be set as high as you need to achieve diminishing returns

happyModel.fit(x=X_train, y=Y_train, epochs=30, batch_size=50)

''' 
--- Note --- 
If you run fit() again, the model continues to train with the parameters it has already learnt instead of reinitializing
------------
'''

# STEP 4 - TEST/EVALUATE THE MODEL
preds = happyModel.evaluate(x=X_test, y=Y_test)
print("\n\nLoss = " + str(preds[0]))
print("\nTest Accuracy = " + str(preds[1]))

'''
If your function worked, you should have observed much better than random-guessing accuracy on the train and test sets

To give you a point of comparison, our model gets around 95% test accuracy in 40 epochs (and 99% train accuracy)
---> Mini batch size of 16
---> "adam" optimizer
---NOTE---
-----> Our model gets decent accuracy after just 2-5 epochs
-----> So if you want to compare diff. models you can train a variety of them for a few epochs and see how they compare


--------------------------------TROUBLESHOOTING TIPS----------------------------------
If you haven't achieved a very good accuracy (more than 80%), here are some tips to try to help you achieve it:
--------------------------------------------------------------------------------------

1. Try using blocks of CONV->BATCHNORM->RELU such as ...
---> X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
---> X = BatchNormalization(axis = 3, name = 'bn0')(X)
---> X = Activation('relu')(X)
---> ... Until your height and width dimensions are quite low and your number of channels quite large (≈32 for example)
---> ... You are encoding useful information in a volume with a lot of channels
---> You can then flatten the volume and use a fully-connected layer.

2. You can use MAXPOOL after the blocks previously desribed
---> It will help you lower the dimension in height and width

3. Change your optimizer (ADAM IS PROBABLY BEST STILL THOUGH)

4. If the model is struggling to run and you get memory issues, lower your batch_size (12 is usually a good compromise)

5. Run on more epochs, until you see the train accuracy plateauing

--------------------------------------------------------------------------------------

Even if you did achieve good accuracy, feel free to keep playing with your model to try to get even better results

---NOTE--- If you perform hyperparameter tuning on your model ---NOTE--- 
------------------------------------------------------------------------
----> The test set actually becomes a dev set, and your model might end up overfitting to the test (dev) set
----> But just for the purpose of this assignment, we won't worry about that here
------------------------------------------------------------------------
'''

'''
CONCLUSION

Congratulations, you have solved the Happy House challenge!

Now, you just need to link this model to the front-door camera of your house
We unfortunately won't go into the details of how to do that here...

What we would like you to remember from this assignment:

--> Keras is a tool we recommend for rapid prototyping
--> It allows you to quickly try out different model architectures
--> Are there any applications of deep learning to your daily life that you'd like to implement using Keras?

 ----------------------------------------------------------------------------------------------------------------
 Remember how to code a model in Keras and the four steps leading to the evaluation of your model on the test set

    1. Create
    2. Compile
    3. Fit/Train
    4. Evaluate/Test
 ----------------------------------------------------------------------------------------------------------------

'''

happyModel.summary()
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))