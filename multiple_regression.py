from utils import *
from ensembles import *
from hypernetworks import *
from typing import Callable
from tqdm import tqdm
import argparse
import pickle
from datetime import datetime


def run_multiple_regression_experiment_PF(
        random_seed: int,                                       # fix random seed
        num_experiments: int,                                   # number of experiments to run
        d: int,                                                 # dimension
        s: int,                                                 # sparsity of ground truths
        n: int,                                                 # number of labeled samples
        N: int,                                                 # number of unlabeled samples
        noise_var: float,                                       # variance of the noise
        cov_stabilizer: float,                                  # stabilizes the condition number of the random covariance matrix A^TA + stab + I -  the smaller, the more adversarial for directly regularized methods
        lambdas: np.ndarray,                                    # array of preference vectors lambda
        dr_ensemble_reg_params: np.ndarray,                     # regularization strengths for directly regularized ensemble
        dr_hypernetwork_regfun: Callable[[float],float],        # regularization strengths for directly regularized hypernetworks, as function lamb -> reg_strength
        ts_ensemble_reg_params: np.ndarray,                     # regularization strengths for two-stage ensemble
        ts_hypernetwork_regfun: Callable[[float],float],        # regularization strengths for two-stage hypernetworks, as function lamb -> reg_strength
        num_epochs: int = 4000,                                 # number of epochs used for hypernetwork training
        verbose: bool = False,                                  # verbosity parameter 
        hn_verbose: bool = False                                # verbosity for hypernetwork training
        ):
    '''
    runs experiments of four described methods.
    returns: dict with error tensors np.ndarray of size [num_experiments, len(lambdas), K]
    '''
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if verbose:
        print('experiment only implemented for K=2 objectives')
    K=2
    gt_errors = np.empty(shape=(num_experiments, len(lambdas), K))
    dr_ensemble_errors = np.empty(shape=(num_experiments, len(lambdas), K))
    dr_hypernetwork_errors = np.empty(shape=(num_experiments, len(lambdas), K))
    ts_ensemble_errors = np.empty(shape=(num_experiments, len(lambdas), K))
    ts_hypernetwork_errors = np.empty(shape=(num_experiments, len(lambdas), K))

    condition_numbers = np.empty(shape=(num_experiments,))

    for exp_ind in tqdm(range(num_experiments)):
        # create random covariances and s-sparse ground-truths
        cov1 = random_covariance_matrix(d, cov_stabilizer)
        cov2 = random_covariance_matrix(d, cov_stabilizer)
        condition_numbers[exp_ind] = np.average((np.linalg.cond(cov1),np.linalg.cond(cov2)))
        if verbose:
            print('condition number of true covariances', np.linalg.cond(cov1), np.linalg.cond(cov2))
        theta1=random_sparse_vector(d,s)
        theta2=random_sparse_vector(d,s)

        # compute gt models
        gt = DirectlyRegularizedEnsemble(
            lambdas=lambdas,
            reg_params=np.zeros(len(lambdas))                       # setting regularization to zero for ground-truth
            )
        gt.train_quadratics(sqrtm(cov1),sqrtm(cov2),sqrtm(cov1)@theta1,sqrtm(cov2)@theta2)  # use the true covariance matrices for ground-truth
        gt.eval_square_loss(theta1,theta2,cov1,cov2)
        gt_errors[exp_ind] = gt.evals

        # sample data
        X1_labeled, X1_unlabeled, y1 = sample_linear_model(theta1,cov1,n,N,noise_var)
        X2_labeled, X2_unlabeled, y2 = sample_linear_model(theta2,cov2,n,N,noise_var)

        # compute directly regularized estimator ensemble
        dr_ensemble = DirectlyRegularizedEnsemble(
            lambdas=lambdas,
            reg_params=dr_ensemble_reg_params
            )
        dr_ensemble.train_quadratics(X1_labeled,X2_labeled,y1,y2)
        dr_ensemble.eval_square_loss(theta1,theta2,cov1,cov2)
        dr_ensemble_errors[exp_ind] = dr_ensemble.evals

        # compute two-stage estimator ensemble
        ts_ensemble = TwoStageEnsemble(
            lambdas=lambdas,
            reg_params=ts_ensemble_reg_params
            )
        ts_ensemble.train_quadratics(X1_labeled,X2_labeled,X1_unlabeled,X2_unlabeled,y1,y2)
        ts_ensemble.eval_square_loss(theta1,theta2,cov1,cov2)
        ts_ensemble_errors[exp_ind] = ts_ensemble.evals

        # compute directly regularized hypernetwork
        dataset = create_hypernetwork_dataset(X1_labeled, X2_labeled, y1, y2)
        dr_hypernetwork = HyperNetwork(K,d)
        train(dr_hypernetwork,num_epochs,dataset,dr_hypernetwork_regfun, verbose=hn_verbose)
        dr_hypernetwork.eval_square_loss(theta1,theta2,cov1,cov2,lambdas)
        dr_hypernetwork_errors[exp_ind] = dr_hypernetwork.evals

        # compute two-stage hypernetwork
        y1_pseudo = X1_unlabeled @ ts_ensemble.expert1
        y2_pseudo = X2_unlabeled @ ts_ensemble.expert2
        dataset = create_hypernetwork_dataset_pseudo(X1_labeled, X2_labeled, X1_unlabeled,X2_unlabeled, y1, y2, y1_pseudo, y2_pseudo)
        ts_hypernetwork = HyperNetwork(K,d)
        train(ts_hypernetwork,num_epochs,dataset,ts_hypernetwork_regfun, verbose=hn_verbose)
        ts_hypernetwork.eval_square_loss(theta1,theta2,cov1,cov2,lambdas)
        ts_hypernetwork_errors[exp_ind] = ts_hypernetwork.evals

    errors = {
        'gt':gt_errors,
        'dr_ensemble':dr_ensemble_errors,
        'dr_hypernetwork':dr_hypernetwork_errors,
        'ts_ensemble':ts_ensemble_errors,
        'ts_hypernetwork':ts_hypernetwork_errors
    }

    return errors, np.average(condition_numbers)


def run_multiple_regression_experiment_unlabeled_sparsity(
        random_seed: int,                               # fix random seed
        num_experiments: int,                           # number of experiments to run
        d:int,                                          # dimension
        s_array: np.ndarray,                            # sparsity of ground truths
        n: int,                                         # number of labeled samples
        N_array: np.ndarray,                            # number of unlabeled samples
        noise_var: float,                               # variance of the noise
        cov_stabilizer: float,                          # stabilizes the condition number of the random covariance matrix A^TA + stab + I -  the smaller, the more adversarial for directly regularized methods
        lambda_fixed: float,                            # fixed preference vectors lambda (only first component)
        ts_ensemble_reg_params: np.ndarray,             # regularization strengths for two-stage ensemble, per sparsity level
        ):
    np.random.seed(random_seed)
    cov1 = random_covariance_matrix(d,cov_stabilizer)
    cov2 = random_covariance_matrix(d,cov_stabilizer)

    errors = {}

    for i in tqdm(range(len(s_array))):

        s = s_array[i]

        theta1 = np.array(s*[1]+(d-s)*[0])/np.sqrt(s)
        theta2 = np.array((d-s)*[0]+s*[1])/np.sqrt(s)

        for N in N_array:
            
            evals = []
            for _ in range(num_experiments):

                # Generate data
                X1_labeled, X1_unlabeled, y1 = sample_linear_model(theta1,cov1,n,N,noise_var)
                X2_labeled, X2_unlabeled, y2 = sample_linear_model(theta2,cov2,n,N,noise_var)
                
                # Perform our estimator with regularization
                ts_ensemble = TwoStageEnsemble(
                    lambdas=[lambda_fixed],
                    reg_params=2*[ts_ensemble_reg_params[i]]
                    )
                ts_ensemble.train_quadratics(X1_labeled,X2_labeled,X1_unlabeled,X2_unlabeled,y1,y2)
                ts_ensemble.eval_square_loss(theta1,theta2,cov1,cov2)
                evals.append(ts_ensemble.evals)
            av_evals = np.average(evals,axis=0)
            
            # Solve population proplem
            gt = DirectlyRegularizedEnsemble(
                lambdas=[lambda_fixed],
                reg_params=[0]                       # setting regularization to zero for ground-truth
                )
            gt.train_quadratics(sqrtm(cov1),sqrtm(cov2),sqrtm(cov1)@theta1,sqrtm(cov2)@theta2)  # use the true covariance matrices for ground-truth
            gt.eval_square_loss(theta1,theta2,cov1,cov2)

            errors[(s, N)] = {
                'ts_loss': av_evals,
                'gt_loss': gt.evals,
            }
    return errors


# run as script

def main():
    parser = argparse.ArgumentParser(description="Run regression experiments.")
    parser.add_argument("--exp", choices=["sparsity-unlabeled", "pareto-fronts"], required=True, help="Choose the experiment to run.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output dictionary.")
    args = parser.parse_args()

    if args.exp == "sparsity-unlabeled":
        errors_matrix = run_multiple_regression_experiment_unlabeled_sparsity(
            random_seed=42,                                                           # fix random seed
            num_experiments=10,                                                      # number of experiments to run
            d=50,                                                                    # dimension
            s_array=np.arange(5, 50, 5),                                             # sparsity of ground truths
            n=15,                                                                    # number of labeled samples
            N_array=np.arange(15, 55, 5),                                            # number of unlabeled samples
            noise_var=0.5,                                                           # variance of the noise
            cov_stabilizer=0.5,                                                      # stabilizes the condition number
            lambda_fixed=0.5,                                                        # fixed preference vectors lambda (only first component)
            ts_ensemble_reg_params=np.array(2 * [1 for s in np.arange(5, 50, 5)]),   # regularization strengths
        )
        with open(args.output+"sparsity-unlabeled"+datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), "wb") as f:
            pickle.dump(errors_matrix, f)


    elif args.exp == "pareto-fronts":
        regs = [1, 5, 7, 8, 8]
        errors_PF, av_cond_number = run_multiple_regression_experiment_PF(
            random_seed=42,                                                   # fix random seed
            num_experiments=10,                                               # number of experiments
            d=100,                                                            # dimension 
            s=1,                                                              # sparsity level
            n=20,                                                             # number of labeled datapoints
            N=200,                                                            # number of unlabeled datapoints
            noise_var=0.5,                                                    # variance of the additive noise
            cov_stabilizer=0.2,                                               # minimum eigenvalue stabilizer
            lambdas=np.linspace(0, 1, 10),                                    # preference weights of the ensemble
            dr_ensemble_reg_params=np.array(regs + list(reversed(regs))),     # reg parameters for direct ensemble
            dr_hypernetwork_regfun=lambda x: 1 + 0.5 * x * (1 - x),           # reg parameters for direct hypernetwork
            ts_ensemble_reg_params=np.array([1, 1]),                          # reg parameters for two-stage ensemble
            ts_hypernetwork_regfun=lambda x: 0,                               # reg parameters for two-stage hypernetwork
            num_epochs=3000,                                                  # number of training epochs
            verbose=False,                                                    # verbosity of training the ensembles
            hn_verbose=False                                                  # verbosity of training the hypernetworks
        )
        with open(args.output+"pareto-fronts"+datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), "wb") as f:
            pickle.dump(errors_PF, f)

if __name__ == "__main__":
    main()

