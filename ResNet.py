import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("\nnumber of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("\nX_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("\nX_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

"""
RESIDUAL NETWORKS (RESNETS)
You will learn how to build very deep convolutional networks, using Residual Networks (ResNets)
In theory, very deep networks can represent very complex functions; but in practice, they are hard to train

Res Nets, introduced by He et al., allow you to train much deeper networks than were previously practically feasible

In this program, you will:

Implement the basic building blocks of ResNets
Put together these building blocks to implement and train a state-of-the-art neural network for image classification
This assignment will be done in Keras

Before jumping into the problem, let's run the cell below to load the required packages (DONE ABOVE)

"""

'''
THE PROBLEM WITH VERY DEEP NEURAL NETWORKS

In recent years, neural networks have become deeper
--> With state-of-the-art networks going from just a few layers (e.g., AlexNet) to over a hundred layers or more

The main benefit of a very deep network is that it can represent very complex functions
It can also learn features at many different levels of abstraction
--> From edges (at the lower layers) to very complex features (at the deeper layers)

However, using a deeper network doesn't always help

A huge barrier to training them is vanishing gradients
--> This is when DNNs have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow
--> Further, during gradient descent, as you backprop through, you are multiplying by the weight matrix on each step
-------> Thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly)

Therefore the magnitude (or norm) of the gradient for the earlier layers decr. to zero very rapidly as training proceeds


In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly back-propagated to earlier layers:

------------------------------------------------------------------------------------------------------------
            
                    ----NORMAL----                                         ----RES NET IDENTITY BLOCK----
                    
            [LAYER] -- [LAYER] -- [LAYER]                                  [LAYER] -- [LAYER] -- [LAYER]
                                                                              |                     |
                                                                              |------CONNECTION-----|
        
        Figure 2 : A ResNet block showing a skip-connection 
            
The image on the left shows the "main path" through the network
The image on the right adds a shortcut to the main path

------------------------------------------------------------------------------------------------------------

By stacking these ResNet blocks on top of each other, you can form a very deep network

Having ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function
--> This means that you can stack on additional ResNet blocks with little risk of harming training set performance
-----> There is also some evidence that the ease of learning an identity function ....
            .... even more than skip connections helping with vanishing gradients ....
            .... accounts for ResNets' remarkable performance

Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dims are same or different
You are going to implement both of them

'''

'''
THERE IS MUCH MORE TEXT INFO ABOUT THIS ON THE JUPYTER NOTEBOOK... TOO MANY PICTURES THOUGH SO SKIPPING
'''

'''
Here're the individual steps.

First component of main path:
----------------------------------------------------------------------------------------------------------
The first CONV2D has F1 filters of shape (1,1) and a stride of (1,1)
--> Its padding is "valid" and its name should be conv_name_base + '2a'
--> Use 0 as the seed for the random initialization

The first BatchNorm is normalizing the channels axis
--> Its name should be bn_name_base + '2a'

Then apply the ReLU activation function
--> This has no name and no hyperparameters
----------------------------------------------------------------------------------------------------------

Second component of main path:
----------------------------------------------------------------------------------------------------------
The second CONV2D has F2 filters of shape (f,f) and a stride of (1,1)
--> Its padding is "same" and its name should be conv_name_base + '2b'
--> Use 0 as the seed for the random initialization.

The second BatchNorm is normalizing the channels axis
--> Its name should be bn_name_base + '2b'

Then apply the ReLU activation function
--> This has no name and no hyperparameters
----------------------------------------------------------------------------------------------------------

Third component of main path:
----------------------------------------------------------------------------------------------------------
The third CONV2D has F3 filters of shape (1,1) and a stride of (1,1)
--> Its padding is "valid" and its name should be conv_name_base + '2c'
--> Use 0 as the seed for the random initialization.

The third BatchNorm is normalizing the channels axis
--> Its name should be bn_name_base + '2c'

NOTE -- There is no ReLU activation function in this component
----------------------------------------------------------------------------------------------------------

Final step:
----------------------------------------------------------------------------------------------------------
The shortcut and the input are added together

Then apply the ReLU activation function
--> This has no name and no hyperparameters
----------------------------------------------------------------------------------------------------------

Exercise:   Implement the ResNet identity block
            We have implemented the first component of the main path
            Please read over this carefully to make sure you understand what it is doing
            You should implement the rest

'''


# GRADED FUNCTION: identity_block

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X


# IMPLEMENT THE RESNET IDENTITY BLOCK
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))

'''
THE CONVOLUTIONAL BLOCK

You've implemented the ResNet identity block
Next, the ResNet "convolutional block" is the other type of block
You can use this type of block when the input and output dimensions don't match up

The difference with the identity block is that there is a CONV2D layer in the shortcut path:

------------------------------------------------------------------------------------------------------------
CONVOLUTIONAL BLOCK
                    
                ---> [CONV2D-->BN-->ReLU] --> [CONV2D-->BN-->ReLU] --> [CONV2D-->BN] ---> [ReLU] --->
                 |                                                                    |
                 |                                                                    |
                 -----------------------------[CONV2D-->BN]----------------------------
       
       
        Figure 2 : A ResNet block showing a skip-connection 

The image on the left shows the "main path" through the network
The image on the right adds a shortcut to the main path

------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------
The CONV2D layer in the shortcut path is used to resize the input X to a different dimension
--> This ensures the dimensions match up in the final addition needed to add the shortcut value back to the main path
------------------------------------------------------------------------------------------------------------
                                                --FOR EXAMPLE--
---> You can use a 1x1 convolution with a stride of 2
---> The CONV2D layer on the shortcut path does not use any non-linear activation function
---> Its main role is to just apply a (learned) linear function that reduces the dimension of the input
---> This results in the dimensions matching up for the later addition step
------------------------------------------------------------------------------------------------------------


************************************************************************************************************
                        The details of the convolutional block are as follows
************************************************************************************************************

First component of main path:
------------------------------------------------------------------------------------------------------------
The first CONV2D has...
---> F1 filters
---> filters of shape (1,1)
---> a stride of (s,s)
---> Its padding is "valid"
---> Its name should be conv_name_base + '2a'

The first BatchNorm is normalizing the channels axis
---> Its name should be bn_name_base + '2a'

Then apply the ReLU activation function
---> This has no name and no hyperparameters
------------------------------------------------------------------------------------------------------------

Second component of main path:
------------------------------------------------------------------------------------------------------------
The second CONV2D has...
---> F2 filters
---> filter shape of (f,f)
---> a stride of (1,1)
---> Its padding is "same"
---> Its name should be conv_name_base + '2b'

The second BatchNorm is normalizing the channels axis
---> Its name should be bn_name_base + '2b'

Then apply the ReLU activation function
---> This has no name and no hyperparameters
------------------------------------------------------------------------------------------------------------

Third component of main path:
------------------------------------------------------------------------------------------------------------
The third CONV2D has...
---> F3 filters
---> filter shape of (1,1)
---> a stride of (1,1)
---> Its padding is "valid"
---> Its name should be conv_name_base + '2c'

The third BatchNorm is normalizing the channels axis
---> Its name should be bn_name_base + '2c'

NOTE: There is no ReLU activation function in this component
------------------------------------------------------------------------------------------------------------

Shortcut path:
------------------------------------------------------------------------------------------------------------
The CONV2D has...
---> F3 filters
---> filter shape of (1,1)
---> a stride of (s,s)
---> Its padding is "valid"
---> Its name should be conv_name_base + '1'

The BatchNorm is normalizing the channels axis
---> Its name should be bn_name_base + '1'

NOTE: There is no ReLU activation function in this component
------------------------------------------------------------------------------------------------------------

Final step:
------------------------------------------------------------------------------------------------------------
The shortcut and the main path values are added together

Then apply the ReLU activation function
---> This has no name and no hyperparameters
------------------------------------------------------------------------------------------------------------

************************************************************************************************************

Exercise:   Implement the convolutional block
            We have implemented the first component of the main path    
            You should implement the rest
            As before, always use 0 as the seed for the random initialization, to ensure consistency

'''


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # **************************************************************
    #                           MAIN PATH
    # **************************************************************

    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same", name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # **************************************************************
    #                        SHORTCUT PATH
    # **************************************************************

    # Parallel component of shortcut
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid", name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # **************************************************************
    #                        PATHS COMBINE
    # **************************************************************

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))

'''
BUILDING YOUR FIRST RESNET MODEL -- 50 LAYERS

You now have the necessary blocks to build a very deep ResNet

The following figure describes in detail the architecture of this neural network
---> "ID BLOCK" in the diagram stands for "Identity block,"
---> "ID BLOCK x3" means you should stack 3 identity blocks together

   [            ]  -->  [              ]  -->  [                ]  -->  [                ]  -->  [                ]  --> 
   [  ZERO PAD  ]  -->  [ {CV-BLK}{MP} ]  -->  [ {CV-BLK}{IDx2} ]  -->  [ {CV-BLK}{IDx3} ]  -->  [ {CV-BLK}{IDx5} ]  -->  
   [            ]  -->  [              ]  -->  [                ]  -->  [                ]  -->  [                ]  -->   

--> [                ]  -->  [                ]  -->  [                ]  -->  [ {OUTPUT} ] 
--> [ {CV-BLK}{IDx2} ]  -->  [ {CV-BLK}{IDx2} ]  -->  [ {AP}{FLAT}{FC} ]  -->  [ {OUTPUT} ]
--> [                ]  -->  [                ]  -->  [                ]  -->  [ {OUTPUT} ]

************************************************************************************************************
                        The details of the resnet-50 model are as follows
************************************************************************************************************

Zero-padding pads the input with a pad of (3,3)
Stage 1:
------------------------------------------------------------------------------------------------------------
The 2D Convolution has...
---> 64 filters
---> filter shape of (7,7)
---> uses a stride of (2,2)
---> Its name is "conv1"

BatchNorm is applied to the channels axis of the input
---> Its name is "bn_conv1"

Activation is applied
---> ReLU

MaxPooling...
---> uses a (3,3) window
---> uses a (2,2) stride
------------------------------------------------------------------------------------------------------------

Stage 2:
------------------------------------------------------------------------------------------------------------
The convolutional block uses...
---> three set of filters
---> filter size of [64,64,256]
---> "f" is 3
---> "s" is 1
---> the block is "a"

The --> 2 <-- identity blocks use...
---> three set of filters
---> filter size of [64,64,256]
---> "f" is 3
---> the blocks are "b" and "c"
------------------------------------------------------------------------------------------------------------

Stage 3:
------------------------------------------------------------------------------------------------------------
The convolutional block uses...
---> three set of filters
---> filter size of [128,128,512]
---> "f" is 3
---> "s" is 2
---> the block is "a"

The --> 3 <-- identity blocks use...
---> three set of filters
---> filter size of [128,128,512]
---> "f" is 3
---> blocks are "b", "c" and "d"
------------------------------------------------------------------------------------------------------------

Stage 4:
------------------------------------------------------------------------------------------------------------
The convolutional block uses...
---> three set of filters
---> filter size of [256, 256, 1024]
---> "f" is 3
---> "s" is 2
---> the block is "a"

The --> 5 <-- identity blocks use... 
---> three set of filters
---> filter size of [256, 256, 1024]
---> "f" is 3 and the blocks are "b", "c", "d", "e" and "f"
------------------------------------------------------------------------------------------------------------

Stage 5:
------------------------------------------------------------------------------------------------------------
The convolutional block uses...
---> three set of filters
---> filter size of [512, 512, 2048]
---> "f" is 3
---> "s" is 2
---> the block is "a"

The --> 2 <-- identity blocks use...
---> three set of filters
---> filter size [512, 512, 2048]
---> "f" is 3
---> the blocks are "b" and "c"

The 2D Average Pooling uses...
---> a window of shape (2,2)
---> its name is "avg_pool"

The flatten doesn't have any hyperparameters or name

The Fully Connected (Dense) layer...
---> reduces its input to the number of classes using a softmax activation
---> Its name should be 'fc' + str(classes)

------------------------------------------------------------------------------------------------------------

************************************************************************************************************

Exercise: Implement the ResNet with 50 layers described in the figure above
          We have implemented Stages 1 and 2
          Please implement the rest
          The syntax for implementing Stages 3-5 should be quite similar to that of Stage 2
          Make sure you follow the naming convention in the text above


--------------------------------------------------------------------------------
You'll need to use this function:
--------------------------------------------------------------------------------
---> Average pooling see reference
--------------------------------------------------------------------------------

Here're some other functions we used in the code below:
--------------------------------------------------------------------------------
---> Conv2D:                  See reference
---> BatchNorm:               See reference (axis: Integer, the axis that should be normalized (typ. features axis))
---> Zero padding:            See reference
---> Max pooling:             See reference
---> Fully conected layer:    See reference
---> Addition:                See reference
--------------------------------------------------------------------------------
'''


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

preds = model.evaluate(X_test, Y_test)

print ("\n\n-------------------------------------------------\n"
       "Loss = " + str(preds[0]))
print ("\n-------------------------------------------------\n"
       "Test Accuracy = " + str(preds[1]))
