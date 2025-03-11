import numpy as np
from scipy.optimize import minimize


# data creation

def random_covariance_matrix(d, stabilizer):
    '''creates a random covariance matrix of for dimension d'''
    A = np.random.randn(d, d)
    cov_matrix = A @ A.T/d  # This ensures the matrix is positive semi-definite
    cov_matrix += stabilizer*np.eye(d)
    return cov_matrix

def random_sparse_vector(d, s):
    '''creates a random s-sparse vector in dimension d, with 2-norm 1'''
    vector = np.zeros(d)
    non_zero_indices = np.random.choice(d, s, replace=False)
    random_values = np.random.randn(s)
    vector[non_zero_indices] = random_values
    vector /= np.linalg.norm(vector)
    return vector

def sample_linear_model(ground_truth,cov,n,N,noise_var):
    d=cov.shape[0]
    X_labeled = np.random.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
    X_unlabeled = np.random.multivariate_normal(mean=np.zeros(d), cov=cov, size=N)
    noise = np.random.normal(size=n,scale=noise_var)
    y = X_labeled @ ground_truth + noise
    return X_labeled, X_unlabeled, y


# optimization

def combined_quadratic(A1,A2,v1,v2,lamb):
    '''creates parameters of sum weighted of two quadratic forms'''
    A = lamb*A1.T@A1 + (1-lamb)*A2.T@A2
    b = lamb*A1.T@v1 + (1-lamb)*A2.T@v2
    return A, b

def solve_quadratic_reg(A,b, reg_strength=0, p=1):
    '''solves the quadratic optimization problem with lp penalty given by 
        min_x x.T A x - 2b.T x + reg_strength |x|_p
    '''
    d = A.shape[0]
    if np.isclose(reg_strength,0.0):
        return np.linalg.pinv(A)@b
    else:
        def objective(x,A,b):
            return x.T@A@x-2*b@x+reg_strength*np.linalg.norm(x,ord=p)
        result = minimize(
            fun=objective,
            x0=np.zeros(d),
            args=(A,b),
            method='L-BFGS-B',
            options={'disp': False}
        )
        return result.x
    
def solve_ls_fair_reg(X,factor, fairnessmatrix,lamb, reg_strength=0, p=1):
    '''solves quadratic optimization problem with l1 penalty'''
    A = lamb*(X.T@X)/X.shape[0] + (1-lamb)*fairnessmatrix
    b = lamb*(X.T@factor)/X.shape[0]
    d = A.shape[0]
    if reg_strength==0:
        sol = np.linalg.pinv(A)@b
    else:
        def objective(f,A,b):
            return f.T@A@f-2*b@f+reg_strength*np.linalg.norm(f,ord=p)
        result = minimize(
            fun=objective,
            x0=np.zeros(d),
            args=(A,b),
            method='L-BFGS-B',  # or another method depending on the problem
            options={'disp': False}
        )
        sol = result.x
    return sol

# losses

def pop_square_loss(x, x_star, cov):
    '''returns population squared loss'''
    diff = x-x_star
    return diff.T@cov@diff

def empirical_square_loss(x,X,y):
    '''returns empirical square loss loss'''
    diff = X@x-y
    return np.inner(diff,diff) / X.shape[0]

def empirical_error_rate(x,X,y):
    '''empirical error rate'''
    return 1 - ((X@x >= 0.5) == y).mean()

def fairness_loss(x,fairnessmatrix):
    '''fairness objective'''
    return np.inner(x,fairnessmatrix@x)