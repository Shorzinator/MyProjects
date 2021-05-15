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
