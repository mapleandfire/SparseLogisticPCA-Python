"""
A demo for the implementation of the Sparse Logistic PCA of the following paper:
    Lee, S., Huang, J. Z., & Hu, J. (2010).
    Sparse logistic principal components analysis for binary data.
    The annals of applied statistics, 4(3), 1579.
"""
from sparse_logistic_pca import SparseLogisticPCA
import numpy as np
import pandas as pd

print("Reading demo data of binary matrix ...")
df = pd.read_csv('./Data/SNPBinaryMatrix.csv', index_col=0,)
dat = df.loc[:, df.sum(axis=0)>0].to_numpy()

print("Creating SparseLogisticPCA model ...")
# n_components is 'k' in the paper
SLPCA = SparseLogisticPCA(n_components=2, verbose=False, max_iters=100)

# automatically finding lambda from BICs - see the paper for details
# skip this step if lambda is known/pre-defined - set it with SLPCA.set_lambda
print("Finding best lambda ...")
lambda_range = np.arange(0.0, 0.00061,0.0006/10)   # the range of lambda needs to be changed based on tasks
best_ldb, BICs, zeros = SLPCA.fine_tune_lambdas(dat, lambdas=lambda_range)

print(f"Setting lambda to {best_ldb}")
SLPCA.set_lambda(best_ldb)

# fitting the model to the data
print("Fitting model ...")
SLPCA.set_n_components(10)  # change n_components (k) if necessary
SLPCA.fit(dat, verbose=True)

# we can obtain the components for inspections
components = SLPCA.get_components()

"""
Notes on the transform function:

        Similar to Sparse PCA, the orthogonality of the learned components is not enforced in Sparse Logistic PCA,
            and hence one cannot use a simple linear projection.
            
        The origin paper does not describe how to transform the new data, and this implementation of transform
            function generally follows that of sklearn.decomposition.SparsePCA:
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
            
        The handling of missing data (N/A) is not supported in this transform implementation.
"""

# get the transformed data
# the missing values are replaced by zeros as the handling of N/A is not supported in this transform implementation
dat_transformed = SLPCA.transform(np.nan_to_num(dat))