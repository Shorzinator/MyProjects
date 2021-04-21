### Importing the libraries
```
import numpy as np
import pandas as pd
import tensorflow as tf
```

## Part 1 - Data Preprocessing
```
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
```

### Encoding categorical data
#### Label Encoding the "Gender" column
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
```
#### One Hot Encoding the "Geography" column
```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
```

### Splitting the dataset into the Training set and Test set
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

### Feature Scaling 
#### (Compulsory in deep learning models)
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Part 2 - Building an ANN
### Initializing the ANN
```
ann = tf.keras.models.Sequential()
```

### Adding input layer and first hidden layer
```
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
```

### Adding second hidden layer
```
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
```

### Adding the output layer
```
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

## Part 3 - Training the ANN
### Compiling the ANN
```
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

### Training the ANN on the training set
```
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
```
