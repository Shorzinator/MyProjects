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
