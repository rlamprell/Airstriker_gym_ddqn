# This is the Convolutional Neural Network
# It links to the DDQN (twice) and is used to select actions and evaluate them
# The configuration below is using the Huber-MaxPooling setup mentioned in the accompanying report

# Package imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


# DISABLED (enabled by default) - Runs faster
tf.compat.v1.disable_eager_execution()


# Build a Neural Network for use within the DDQN model
# - Changes could be made to pass in every variable, however, to make it easier to read some things were left static
def build_nn(lr, n_actions, frame_height, frame_width, frame_layers, model_name):

    # Create a neural network with these layers
    model = Sequential([

        # Input shape - doesn't actually make a layer
        Input(shape=(frame_width, frame_height, frame_layers),  name='Input_Shape'),

        # Convolution layers to search for shapes and shrink image size
        # - The final Conv2D layer is likely not doing much in this config
        Conv2D(16, (4, 4), strides=(1, 1), activation='relu',   name='1st_Convolution'),
        Conv2D(32, (4, 4), strides=(2, 2), activation='relu',   name='2nd_Convolution'),
        Conv2D(32, (2, 2), strides=(1, 1), activation='relu',   name='3rd_Convolution'),

        # Max pooling to further shrink image size - Excluded from Initial network
        MaxPool2D( (2, 2), strides=None, padding='Valid',       name='Max_Pooling'),

        # Flatten to a a single array so that the information can be used in a Dense layer
        Flatten(name='Flattener'),

        # Fully connected layers
        Dense(256, activation='relu',                           name='Fully_Connected_1'),
        Dense(256, activation='relu',                           name='Fully_Connected_2'),

        # Output layer containing the available actions (the fewer the better)
        Dense(n_actions,                                        name='Output')], name=model_name)

    # Compile the model using the layers above, the Adam optimiser and a loss function
    # Originally used loss='mse'
    model.compile(optimizer=Adam(lr=lr), loss=Huber())

    # Print a summary of the model to he console
    model.summary()

    return model