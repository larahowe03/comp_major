# Importing Keras 
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

# Parameters
# ===============================
DATASET_DIR = "final_dataset"   
IMG_SIZE = 64                   
RANDOM_SEED = 7

xdata = []
ydata = []


piece_to_label = {}
label_counter = 0

# Load in the dataset and label the ouputs based on folder names 
for class_name in os.listdir(DATASET_DIR):
    if class_name.startswith('.'):
        continue  # skip hidden stuff like .DS_Store

    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    
    if class_name not in piece_to_label:
        piece_to_label[class_name] = label_counter
        label_counter += 1

    label = piece_to_label[class_name]

    # Loop through images inside the class folder
    for image_name in os.listdir(class_dir):
        if image_name.startswith('.'):
            continue

        img_path = os.path.join(class_dir, image_name)

        # Read as grayscale (1 channel)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        # Resize to a fixed size
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        xdata.append(img_resized)
        ydata.append(label)


# Convert to numpy arrays
xdata = np.array(xdata)
ydata = np.array(ydata)

print(f"Loaded {len(xdata)} images")
print("Class mapping (folder -> label):")
for k, v in piece_to_label.items():
    print(f"  {k}: {v}")

img_rows, img_cols = xdata.shape[1], xdata.shape[2]
print(f"Resized image dimensions: {img_rows}x{img_cols}")

# Test train split 
x_train, x_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.3, random_state=RANDOM_SEED, stratify=ydata
)

print(len(x_train), "train images,", len(x_test), "test images")

# Reshape 
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test  = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test  = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Normalise 
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# One-hot encode labels
num_classes = len(piece_to_label)
print(f"Number of classes: {num_classes}")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print("input_shape for CNN:", input_shape)


########## Train the model #####################
# ----- Data augmentation block -----
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(0.03, 0.03),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# ----- Model: 3 conv (32,64,128), Dense 256, WITH augmentation ---
model = Sequential()

# Input + augmentation
model.add(layers.Input(shape=input_shape)) 
model.add(data_augmentation)

# Conv block 1
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv block 2 (3rd conv)
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Classifier head
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))   # Dense 256
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# setup loss function (cross-entropy) and optimiser (adam)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy']) 
print(model.summary())


batch_size = 128
epochs = 100

early_stop = EarlyStopping(
    monitor='val_loss',    
    patience=10,            
    restore_best_weights=True
)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split = 0.2,
          callbacks = [early_stop],
          shuffle = True,
          verbose=1,
          )

# Evaluate trained model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Make predictions on the test data
predictions = model.predict(x_test)
predictions = np.argmax(predictions,axis=1)

labels = np.argmax(y_test , axis=1)