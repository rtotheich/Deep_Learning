#Import the modules required for the Deep Learning and math in this example

import tensorflow as tf
import numpy as np
from tensorflow import keras

#Creates a sequential network with a single neuron and one layer deep

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Initiates an optimizer and loss function with stochastic gradient descent
# for optimizer and mean squared error for loss

model.compile(optimizer='sgd', loss='mean_squared_error')

# Instantiate parallel arrays with x and y values

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the model by asking the computer to evaluate all the possible
# combinations between the x and y values by passing them
# through the net 500 times (epochs) in order to guess the relationship
# between the values. The correct solution is y = 3x + 1.

model.fit(xs, ys, epochs=500)

# Print the model's estimated value when plugged with the value 10.

print(model.predict([10.0]))
