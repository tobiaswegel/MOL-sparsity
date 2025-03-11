from preprocessing import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from datetime import datetime
import argparse
from ensembles import *
import warnings
warnings.filterwarnings("ignore")

def compute_fairnessmatrix(X, protected_index):
    '''computes fairness matrix to compute fairness metrix as a quadratic form'''
    threshold = 0
    group_1=(X[:,protected_index]>=threshold)
    group_0=(X[:,protected_index]<threshold)
    mean_1 = np.average(X[group_1],axis=0)
    mean_0 = np.average(X[group_0],axis=0)
    M = 0.25*np.outer(mean_0-mean_1,mean_0-mean_1)
    return M


def run_experiments_fairness_PF(
        random_seed: int,
        num_experiments: int,
        dataset_list: list,
        lambdas = np.linspace(0,1,10),
        verbose = False
    ):

    np.random.seed(random_seed)

    errors = {}

    for dataset in dataset_list:
        print("\n running experiment on:",dataset["name"],"\n")

        K=2
        gt_errors = np.empty(shape=(num_experiments, len(lambdas), K))
        dr_ensemble_errors = np.empty(shape=(num_experiments, len(lambdas), K))
        ts_ensemble_errors = np.empty(shape=(num_experiments, len(lambdas), K))

        X,y = load_preprocess(dataset["name"],verbose=verbose,num_noisy_feats=dataset["num_noisy_feats"])
        train_size = dataset["train_size"]
        test_size = dataset["test_size"]
        labeled_size = dataset["labeled_size"]
        unlabeled_size = train_size - labeled_size
        protected_index = dataset["protected_index"]
        eval_metric = dataset['eval_metric']
        
        for exp_ind in tqdm(range(num_experiments)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)#, random_state=2
            X_labeled, X_unlabeled , y_train, _ = train_test_split(X_train, y_train, train_size=labeled_size, test_size=unlabeled_size)
            train_fairnessmatrix = compute_fairnessmatrix(X_train,protected_index)
            test_fairnessmatrix = compute_fairnessmatrix(X_test,protected_index)

            # compute ground-truth
            gt = DirectlyRegularizedEnsemble(
                lambdas=lambdas,
                reg_params=np.zeros(len(lambdas))                       # setting regularization to zero for ground-truth
                )
            gt.train_quadratic_fairness(X_test,y_test,test_fairnessmatrix)
            gt.eval_accuracy_fairness(X_test,y_test,test_fairnessmatrix, metric=eval_metric)
            gt_errors[exp_ind] = gt.evals
            
            # compute directly regularized estimator ensemble
            dr_ensemble = DirectlyRegularizedEnsemble(
                lambdas=lambdas,
                reg_params=dataset["dr_ensemble_reg_params"]
                )
            dr_ensemble.train_quadratic_fairness(X_labeled,y_train,train_fairnessmatrix)
            dr_ensemble.eval_accuracy_fairness(X_test,y_test,test_fairnessmatrix, metric=eval_metric)
            dr_ensemble_errors[exp_ind] = dr_ensemble.evals

            # compute two-stage estimator ensemble
            ts_ensemble = TwoStageEnsemble(
                lambdas=lambdas,
                reg_params=dataset["ts_ensemble_reg_params"]
                )
            ts_ensemble.train_quadratic_fairness(X_labeled,X_unlabeled,y_train,train_fairnessmatrix)
            ts_ensemble.eval_accuracy_fairness(X_test,y_test,test_fairnessmatrix, metric=eval_metric)
            ts_ensemble_errors[exp_ind] = ts_ensemble.evals
    
        errors[dataset["name"]] = {"gt": None, "dr": None, "ts": None}
        errors[dataset["name"]]["gt"] = gt_errors
        errors[dataset["name"]]["dr_ensemble"] = dr_ensemble_errors
        errors[dataset["name"]]["ts_ensemble"] = ts_ensemble_errors

    return errors
            

def main():
    parser = argparse.ArgumentParser(description="Run fairness experiments and save errors.")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the errors file."
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", choices=["communities", "adult", "hsls", "enem"], required=True,
        help="List of datasets to include in the experiment."
    )
    parser.add_argument(
        "--num_experiments", type=int, default=20, help="Number of experiments to run (default: 20)."
    )
    args = parser.parse_args()

    communities = {
        "name": "communities",
        "protected_index": 7,
        "train_size": 1000,
        "test_size": 1000,
        "labeled_size": 150,
        "num_noisy_feats": 0,
        "dr_ensemble_reg_params": 10 * [0.04],
        "ts_ensemble_reg_params": [0.04, 0.04],
        "eval_metric": "square_loss"
    }
    adult = {
        "name": "adult",
        "protected_index": 9,
        "train_size": 10_000,
        "test_size": 30_000,
        "labeled_size": 500,
        "num_noisy_feats": 100,
        "dr_ensemble_reg_params": 10 * [0.01],
        "ts_ensemble_reg_params": [0.01, 0.01],
        "eval_metric": "error_rate"
    }
    hsls = {
        "name": "hsls",
        "protected_index": 57,
        "train_size": 5000,
        "test_size": 5000,
        "labeled_size": 1000,
        "num_noisy_feats": 0,
        "dr_ensemble_reg_params": 10 * [0.01],
        "ts_ensemble_reg_params": [0.01, 0.01],
        "eval_metric": "error_rate"
    }
    enem = {
        "name": "enem",
        "protected_index": 1,
        "train_size": 10_000,
        "test_size": 5_000,
        "labeled_size": 2000,
        "num_noisy_feats": 0,
        "dr_ensemble_reg_params": 10 * [0.01],
        "ts_ensemble_reg_params": [0.01, 0.01],
        "eval_metric": "error_rate"
    }
    
    dataset_dict = {
        "communities": communities,
        "adult": adult,
        "hsls": hsls,
        "enem": enem
    }

    # Filter the datasets based on argparse input
    dataset_list = [dataset_dict[name] for name in args.datasets]

    errors = run_experiments_fairness_PF(
        random_seed=42,                                   # fix the random seed
        num_experiments=args.num_experiments,                               # number of experiments
        dataset_list=dataset_list,                       # datasets included in the experiment
        lambdas=np.linspace(0, 1, 10),                   # preference weights of the ensembles
        verbose=False                                    # verbosity
    )

    with open(args.output+"fairness"+datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), "wb") as f:
        pickle.dump(errors, f)

if __name__ == "__main__":
    main()


