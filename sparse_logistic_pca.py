import numpy as np
from sklearn.linear_model import ridge_regression

def inv_logit_mat(x, min=0.0, max=1.0,):
    # the inverse logit transformation function
    p = np.exp(x) / (1.0+np.exp(x))
    which_large = np.isnan(p) & (~np.isnan(x))
    p[which_large] = 1.0
    return p*(max-min)+min

def sparse_logistic_pca(
        dat, lbd=0.0006, k=2, verbose=False, max_iters=100, crit=1e-5,
        randstart=False, procrustes=True, lasso=True,
):
    """
    A Python implementation of the sparse logistic PCA of the following paper:
        Lee, S., Huang, J. Z., & Hu, J. (2010). Sparse logistic principal components analysis for binary data.
        The annals of applied statistics, 4(3), 1579.

    This implementation is migrated from this R package:
        https://github.com/andland/SparseLogisticPCA

    Args:
        dat: input data, n*d numpy array where n is the numbers of samples and d is the feature dimensionality
        lbd: the lambda value, higher value will lead to more sparse components
        k: the dimension after reduction
        verbose: print log or not
        max_iters: maximum number of iterations
        crit: the minimum difference criteria for stopping training
        randstart: randomly initialize A, B, mu or not
        procrustes: procrustes
        lasso: whether to use LASSO solver

    Returns: a dict containing the results

    """

    ### Initialize q
    q = 2*dat-1
    q[np.isnan(q)] = 0.0
    n,d = dat.shape

    ### Initialize mu, A, B
    if not randstart:
        mu = np.mean(q,axis=0)
        udv_u, udv_d, udv_v = np.linalg.svd(q-np.mean(q,axis=0), full_matrices=False)
        A = udv_u[:,0:k].copy()
        B = np.matmul(udv_v[0:k,:].T, np.diag(udv_d[0:k]))
    else:
        mu = np.random.normal(size=(d,))
        A = np.random.uniform(low=-1.0, high=1.0, size=(n,k,))
        B = np.random.uniform(low=-1.0, high=1.0, size=(d,k,))

    loss_trace = dict()

    ## loop to optimize the loss, see Alogrithm 1 in the paper
    for m in range(max_iters):

        last_mu, last_A, last_B = mu.copy(), A.copy(), B.copy()

        theta = np.outer(np.ones(n), mu) + np.matmul(A, B.T)
        X = theta+4*q*(1-inv_logit_mat(q*theta))
        Xcross = X - np.matmul(A, B.T)
        mu = np.matmul((1.0/n) * Xcross.T, np.ones(n))

        theta = np.outer(np.ones(n), mu) + np.matmul(A, B.T)
        X = theta+4*q*(1-inv_logit_mat(q*theta))
        Xstar = X-np.outer(np.ones(n), mu)

        if procrustes:
            M_u, M_d, M_v = np.linalg.svd(np.matmul(Xstar, B), full_matrices=False)
            A = np.matmul(M_u, M_v)
        else:
            A = Xstar @ B @ np.linalg.inv(B.T @ B)
            A, _ = np.linalg.qr(A)

        theta = np.outer(np.ones(n), mu) + A @ B.T
        X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
        Xstar = X-np.outer(np.ones(n), mu)

        if lasso:
            B_lse = Xstar.T @ A
            B = np.sign(B_lse) * np.maximum(0.0, np.abs(B_lse)-4*n*lbd)
        else:
            C = Xstar.T @ A
            B = (np.abs(B) / (np.abs(B)+4*n*lbd)) * C

        q_dot_theta = q*(np.outer(np.ones(n),mu) + A @ B.T)
        loglike = np.sum(np.log(inv_logit_mat(q_dot_theta))[~np.isnan(dat)])
        penalty = n*lbd*np.sum(abs(B))
        loss_trace[str(m)] = (-loglike+penalty) / np.sum(~np.isnan(dat))

        if verbose:
            print(f"Iter: {m} - Loss: {loss_trace[str(m)]:.4f}, NegLogLike: {-loglike:.4f}, Penalty: {penalty:.4f} ")

        if m>3:
            if loss_trace[str(m-1)] - loss_trace[str(m)] < crit:
                break

    if loss_trace[str(m-1)] < loss_trace[str(m)]:
        mu, A, B, m = last_mu, last_A, last_B, m-1

        q_dot_theta = q*(np.outer(np.ones(n),mu) + A @ B.T)
        loglike = np.sum(np.log(inv_logit_mat(q_dot_theta))[~np.isnan(dat)])

    zeros = sum(np.abs(B))
    BIC = -2.0*loglike+np.log(n)*(d+n*k+np.sum(np.abs(B)>=1e-10))

    res = {
        "mu":mu, "A":A, "B":B, "zeros":zeros,
        "BIC":BIC, "iters":m, "loss_trace":loss_trace, "lambda":lbd,
    }

    return res

class SparseLogisticPCA(object):
    """
    A warper class of sparse logistic PCA, which provides the fit, transform and fit_transform methods
    """
    def __init__(
            self, lbd=0.0001, n_components=2, verbose=False, max_iters=100, crit=1e-5,
            randstart=False, procrustes=True, lasso=True,
            ridge_alpha=0.01,
    ):
        """
        Args:
            lbd: the lambda value, higher value will lead to more sparse components
            n_components: the dimension after reduction, i.e. k in the origin paper
            verbose: print log or not
            max_iters: maximum number of iterations
            crit: the minimum difference criteria for stopping training
            randstart: randomly initialize A, B, mu or not
            procrustes: procrustes
            lasso: whether to use LASSO solver
            ridge_alpha: Amount of ridge shrinkage to apply in order to improve conditioning when
                calling the transform method.
        """
        self.lbd = lbd
        self.n_components = n_components
        self.verbose=verbose
        self.max_iters = max_iters
        self.crit = crit
        self.randstart = randstart
        self.procrustes = procrustes
        self.lasso=lasso
        self.ridge_alpha = ridge_alpha

    def fit(self, dat, verbose=False):
        """

        Args:
            dat: ndarray of shape (n_samples, n_features), the data to be fitted
            verbose: print log or not

        Returns:
            self

        """
        res = sparse_logistic_pca(
            dat, lbd=self.lbd, k=self.n_components, verbose=verbose,
            max_iters=self.max_iters, crit=self.crit,
            randstart=self.randstart, procrustes=self.procrustes,
            lasso=self.lasso,)

        self.mu, self.components_ = res['mu'], res['B'].T
        _, self.d = dat.shape

        components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm

        return self

    def transform(self, X):
        """

        Similar to Sparse PCA, the orthogonality of the learned components is not enforced in Sparse Logistic PCA,
            and hence one cannot use a simple linear projection.

        The origin paper does not describe how to transform the new data, and this implementation of transform
            function generally follows that of sklearn.decomposition.SparsePCA:
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html

        The handling of missing data (N/A) is not supported in this transform implementation.

        Args:
            X: ndarray of shape (n_samples, n_features), the input data

        Returns:
            ndarray of (n_samples, n_components), the data after dimensionality reduction

        """
        n, d = X.shape
        assert d==self.d,\
            f"Input data should have a shape (n_samples, n_features) and n_features should be {self.d}"

        Xstar = X - np.outer(np.ones(n), self.mu)

        U = ridge_regression(
            self.components_.T, Xstar.T, self.ridge_alpha, solver="cholesky",
        )

        return U

    def fit_transform(self, dat):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Args:
            dat: ndarray of shape (n_samples, n_features)

        Returns:
            ndarray of (n_samples, n_components)

        """
        self.fit(dat)
        return self.transform(dat)

    def fine_tune_lambdas(self, dat, lambdas=np.arange(0, 0.00061, 0.0006 / 10)):
        # fine tune the Lambda values based on BICs following the paper
        BICs, zeros = [], []
        for lbd in lambdas:
            # print(f"Lambda: {lbd:.6f}")
            this_res = sparse_logistic_pca(
            dat, lbd=lbd, k=self.n_components, verbose=False,
            max_iters=self.max_iters, crit=self.crit,
            randstart=self.randstart, procrustes=self.procrustes,
            lasso=self.lasso,)
            BICs.append(this_res['BIC'])
            zeros.append(this_res['zeros'])
        best_ldb = lambdas[np.argmin(BICs)]
        return best_ldb, BICs, zeros

    def set_lambda(self,new_lbd):
        print(f"Setting lambda to: {new_lbd}")
        self.lbd = new_lbd

    def set_ridge_alpha(self, ridge_alpha):
        self.ridge_alpha = ridge_alpha

    def set_n_components(self, n_components):
        self.n_components = n_components

    def get_components(self):
        return self.components_