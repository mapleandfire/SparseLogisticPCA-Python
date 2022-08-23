
# Sparse Logistic PCA
An unofficial Python implementation of the Sparse Logistic PCA of the following paper:
```
Lee, S., Huang, J. Z., & Hu, J. (2010). 
Sparse logistic principal components analysis for binary data. 
The annals of applied statistics, 4(3), 1579.
```
The implementation of the algorithm follows this [R package](https://github.com/andland/SparseLogisticPCA). 

## Dependencies
`numpy` `scikit-learn` (`pandas` is also required to read csv in the demo)

## Usage
A minimal example:
```
from sparse_logistic_pca import SparseLogisticPCA
import numpy as np

# the binary matrix to fit, shape: (n_samples, n_features)
dat = np.random.randint(2, size=(40,16,),).astype(float)  

# create model and fit data
# lbd: lambda controlling sparsity of components
SLPCA = SparseLogisticPCA(n_components=2, lbd=0.0001)  
SLPCA.fit(dat, verbose=True)  

# the binary matrix to transform, shape: (n_samples, n_features)
X = np.random.randint(2, size=(20,16,),).astype(float)  

# get the transformed data
X_t = SLPCA.transform(X)  
```
See `demo.py` for an example on how to fine-tune the Lambda values. 
