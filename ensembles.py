import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from utils import *


class Ensemble():
    def __init__(self, lambdas, reg_params):
        self.lambdas = lambdas
        self.reg_params = reg_params
        self.ensemble_size = len(lambdas)
        self.model_params = None
        self.evals = None

    def eval_square_loss(self,theta1,theta2,cov1,cov2):
        losses = np.empty(shape=(self.ensemble_size,2))
        for i in range(self.ensemble_size):
            params = self.model_params[i]
            losses[i,0] = pop_square_loss(params,theta1,cov1)
            losses[i,1] = pop_square_loss(params,theta2,cov2)
        self.evals = losses

    def eval_accuracy_fairness(self,X,y,fairness_matrix,metric='error_rate'):
        losses = np.empty(shape=(self.ensemble_size,2))
        for i in range(self.ensemble_size):
            params = self.model_params[i]
            if metric == 'error_rate':
                losses[i,0] = empirical_error_rate(params,X,y)
            elif metric == 'square_loss':
                losses[i,0] = empirical_square_loss(params,X,y)
            else:
                raise RuntimeError('metric not implemented')
            losses[i,1] = fairness_loss(params,fairness_matrix)
        self.evals = losses


class DirectlyRegularizedEnsemble(Ensemble):

    def __init__(self, lambdas, reg_params):
        super().__init__(lambdas,reg_params)
        if not self.ensemble_size == len(reg_params):
            raise RuntimeError('number of reg_params and lambdas is not equal')
    
    def train_quadratics(self, X1, X2, y1, y2):
        d = X1.shape[1]
        self.model_params=np.empty(shape=(self.ensemble_size,d))
        for i in range(self.ensemble_size):
            A, b = combined_quadratic(X1, X2, y1, y2, self.lambdas[i])
            sol = solve_quadratic_reg(A,b,self.reg_params[i],p=1)
            self.model_params[i] = sol

    def train_quadratic_fairness(self, X, y, fairness_matrix):
        d = X.shape[1]
        self.model_params=np.empty(shape=(self.ensemble_size,d))
        for i in range(self.ensemble_size):
            sol = solve_ls_fair_reg(X,y,fairness_matrix, self.lambdas[i],self.reg_params[i],p=1)
            self.model_params[i] = sol

class TwoStageEnsemble(Ensemble):

    def __init__(self, lambdas, reg_params):
        super().__init__(lambdas,reg_params)
        if not 2 == len(reg_params):
            raise RuntimeError('number of reg_params is not 2')
        self.expert1 = None
        self.expert2 = None

    def train_quadratics(self,X1_labeled,X2_labeled,X1_unlabeled,X2_unlabeled,y1,y2):
        d = X1_labeled.shape[1]
        self.model_params=np.empty(shape=(self.ensemble_size,d))
        
        # stage 1: train expert models
        A1 = X1_labeled.T @ X1_labeled
        b1 = X1_labeled.T @ y1
        thetahat1 = solve_quadratic_reg(A1,b1,reg_strength=self.reg_params[0])
        self.expert1=thetahat1
        A2 = X2_labeled.T @ X2_labeled
        b2 = X2_labeled.T @ y2
        thetahat2 = solve_quadratic_reg(A2,b2,reg_strength=self.reg_params[1])
        self.expert2=thetahat2
       
        # stage 2: optimize
        X1=np.concatenate((X1_labeled,X1_unlabeled),axis=0) if X1_unlabeled.size else X1_labeled
        X2=np.concatenate((X2_labeled,X2_unlabeled),axis=0) if X2_unlabeled.size else X2_labeled
        for i in range(self.ensemble_size):
            A, b = combined_quadratic(X1, X2, X1@thetahat1, X2@thetahat2, self.lambdas[i])
            sol = solve_quadratic_reg(A,b,0.0,p=1)
            self.model_params[i] = sol

    def train_quadratic_fairness(self, X_labeled, X_unlabeled, y, fairness_matrix):
        d = X_labeled.shape[1]
        self.model_params=np.empty(shape=(self.ensemble_size,d))
        
        # stage 1: train expert models
        thetahat = solve_ls_fair_reg(X_labeled, y, fairness_matrix, 1, self.reg_params[0], p=1)
        self.expert1=thetahat
       
        # stage 2: optimize
        X = np.concatenate((X_labeled,X_unlabeled),axis=0) if X_unlabeled.size else X_labeled
        for i in range(self.ensemble_size):
            sol = solve_ls_fair_reg(X, X@thetahat, fairness_matrix, self.lambdas[i], 0, p=1)
            self.model_params[i] = sol
            

if __name__ == '__main__':
    raise RuntimeError('no main')