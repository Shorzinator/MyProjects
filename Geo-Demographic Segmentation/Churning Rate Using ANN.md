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
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))     # if the dependant variable was non-binary, activation = 'softmax'
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

## Part 4 - Making the predictions and evaluating the model
### Predicting the result of a single observation

Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: Spain

Credit Score: 588

Gender: Male

Age: 40 years old

Tenure: 3 years

Balance: \$ 70000

Number of Products: 3

Does this customer have a credit card? Yes

Is this customer an Active Member: Yes

Estimated Salary: \$ 40000

So, should we say goodbye to that customer?

**Solution**
```
print(ann.predict(sc.transform([[0, 0, 1, 588, 1, 40, 3, 70000, 3, 1, 1, 40000]])) > 0.5)
```
```
[[True]]
```
Therefore, our ANN model predicts that this customer exits the bank!


**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.

**Important note 2:** Notice also that the "Spain" country was not input as a string in the last column but as "0, 0, 1" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "Spain" was encoded as "0, 0, 1". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.


### Predicting the Test set results
```
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

### Making the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy = ", accuracy_score(y_test, y_pred) * 100, "%")
```

The accuracy comes out to be - 
```
[[1515   80]
 [ 201  204]]
Accuracy = 85.95 %          
```
