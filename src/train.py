import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model  

X = np.load("X.npy")
Y = np.load("Y.npy")
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y,
    test_size= 0.2,
    random_state=42,
    stratify=Y
)
print("Training samples: ", X_train.shape)
print("Validation samples: ", X_val.shape)

weights_path = "data/weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
base_model = MobileNetV2( 
    include_top=False,
    weights =weights_path,
    input_shape =(224, 224, 3)
)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output= Dense(1, activation="sigmoid")(x)
model = Model(inputs= base_model.input, outputs=output)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics= ["accuracy"])
model.summary()
history = model.fit(
    X_train, Y_train,
    validation_data= (X_val, Y_val),
    epochs = 5,
    batch_size=8
)
model.save("drowsiness_model.h5")
print("Model saved successfully!")

