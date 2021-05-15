## Data Preprocessing
### Importing the libraries
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### Importing the training set
```
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv)
training_set = dataset_train.iloc[:, 1:2].values
```

### Feature Scaling
```
from sklearn.preprocessing import MinMaxScalar
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

### Creating a data sctructure with 60 timesteps and 1 output 
#### The '60 timesteps' count is experimentally determined
```
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)    
```

### Reshaping
```
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

## Building the RNN
### Importing the Keras libraries and packages
```
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
```

### Initializing the RNN
```
regressor = Sequential()
```

### Adding the first LSTM layer and some Dropout regularisation
```
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
```

### Adding a second LSTM layer and some Dropout regularisation
```
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
```

### Adding a third LSTM layer and some Dropout regularisation
```
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
```

### Adding a fourth LSTM layer and some Dropout regularisation
```
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
```

### Adding the output layer
```
regressor.add(Dense(units = 1))
```

### Compiling the RNN
```
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
```

### Fitting the RNN to the Training set
```
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```
