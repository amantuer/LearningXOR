import tensorflow as tf
import numpy as np

# XOR input and output
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(2,), activation='relu'), # Hidden layer with 8 units
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=10000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, Y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Predict
predictions = model.predict(X)
print("Predictions:")
print(predictions)


"""
#old code not working well
#1/1 [==============================] - 0s 106ms/step
#[[0.49999997]
#[0.49999997]
#[0.49999997]
#[0.5       ]]

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# XOR data
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Model architecture
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10000, verbose=0)

# Evaluate the model
predictions = model.predict(X)
print(predictions)
"""