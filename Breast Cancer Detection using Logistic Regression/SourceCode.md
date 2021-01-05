### Importing the libraries
```
import pandas as pd
```

### Importing the dataset
```
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

