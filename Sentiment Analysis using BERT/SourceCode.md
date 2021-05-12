### Importing libraries
```
pip3 install ktrain   # used to implement BERT
```
```
import os.path
import numpy as np
import tensorflow as tf
import ktrain
from ktrain import text
```
## Data Preprocessing
### Loading the IMDB Dataset
```
dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz",
                                  origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                  extract=True)
IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')
```
