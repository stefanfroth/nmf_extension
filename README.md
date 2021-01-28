# nmf_extension

This repository extends the functionality of the class sklearn.decomposition.NMF


### Installation

```
pip install git+https://github.com/stefanfroth/nmf_extension
```


### Usage

```
import numpy as np
import pandas as pd
form sklearn.datasets import load_digits
from nmf_extension.nmf import CustomNMF

X, y = load_digits(return_X_y=True)

# set some values to nan
mask = np.random.randint(0, 2, size=X.shape).astype(np.bool) 
X[mask] = np.nan

# run the model
m = CustomNMF(n_components=20) 
m.fit_transform(X)
```
