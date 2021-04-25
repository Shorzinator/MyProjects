# Convolutional Neural Network
### Importing Libraries
```
import tensorflow as tf
from keras.preprocessing.image import ImageDateGenerator
```

## Part 1 - Data Preprocessing
### Preprocessing the Training set
```
train_datagen = ImageData Generator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)  # parameters on which the training data would be processed
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
```

### Preprocessing the Test Data
```
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directroy('dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
```

## Part 2 - Building the CNN
### Initialising
```
cnn = tf.keras.models.Sequential()
```

### First Convolutional Layer and Pooling Layer
```
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))  # Convolutional Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))  # Pooling Layer
```

### Second Convolutional Layer and Pooling Layer
```
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
```
## Step 3 - Flattening
```
cnn.add(tf.keras.layers.Flattten())
```

## Step 4 - Fully Connected Layer
```
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
```

## Step 5 - Output Layer
```
cnn.add(tf.keras.layers.Dence(units = 1, activation = 'sigmoid'))
```
