# leave-one-out-encoder
Leave one out coding for categorical features

See the source for this project here:
<https://github.com/welfare520/leave-one-out-encoder>.

## Getting Started  

### Installing

```
$ pip install loo_encoder
```

## Example

Fit encoder according to X and y, and then transform it.
 
```python
from loo_encoder.encoder import LeaveOneOutEncoder
import pandas as pd
import numpy as np


enc = LeaveOneOutEncoder(cols=['gender', 'country'], handle_unknown='impute', sigma=0.02, random_state=42)

X = pd.DataFrame(
    {
        "gender": ["male", "male", "female", "male"],
        "country": ["Germany", "USA", "USA", "UK"],
        "clicks": [10, 33, 47, 21]
    }
)

y = pd.Series([150, 250, 300, 100], name="orders")

df_train = enc.fit_transform(X=X, y=y, sample_weight=X['clicks'])
```


Perform the transformation to new categorical data.

```python
X_val = pd.DataFrame(
    {
        "gender": ["unknown", "male", "female", "male"],
        "country": ["Germany", "USA", "Germany", "Japan"]
    }
)

df_test = enc.transform(X=X_val)
```
