import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def add_noisy_features(X, num=0):
    return np.concatenate([X, np.random.normal(0, 1, (X.shape[0], num))], axis=1) if num > 0 else X

def load_preprocess(name,verbose=False,num_noisy_feats=0):
    if name == "communities":
        return load_preprocess_communities(verbose=verbose, num_noisy_feats=num_noisy_feats)
    elif name == "adult":
        return load_preprocess_adult(verbose=verbose, num_noisy_feats=num_noisy_feats)
    elif name == "hsls":
        return load_preprocess_HSLS(verbose=verbose, num_noisy_feats=num_noisy_feats)
    elif name == "enem":
        return load_preprocess_ENEM(verbose=verbose, num_noisy_feats=num_noisy_feats)
    else:
        raise RuntimeError("Unknown dataset")
    

def load_preprocess_communities(verbose=False, num_noisy_feats=0):
    df = pd.read_csv("data/crimedata.csv", sep=',', encoding='latin-1', na_values="?")
    df.rename(columns={'Êcommunityname':'communityName'}, inplace=True)
    cols = ['HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded',
            'PctVacMore6Mos','PctUnemployed','PctEmploy','murdPerPop',
            'rapesPerPop','robbbPerPop','assaultPerPop','ViolentCrimesPerPop',
            'burglPerPop','larcPerPop','autoTheftPerPop','arsonsPerPop','nonViolPerPop']
    df[cols].fillna(df[cols].median(), inplace=True)
    df = df.apply(LabelEncoder().fit_transform)
    df = (df - df.mean()) / df.std()
    if verbose:
        print(df.head(2))
    datamatrix = df.values
    X, y, y_nonviolent = datamatrix[:, :-2], datamatrix[:, -2], datamatrix[:, -1]
    X = add_noisy_features(X,  num_noisy_feats)
    if verbose:
        print(f'Number of samples: {X.shape[0]}, Dimension: {X.shape[1]}')
    return X, y

def load_preprocess_adult(verbose=False, num_noisy_feats=0):
    df = pd.read_csv("data/adult.csv")
    enc = LabelEncoder()
    df = df.apply(enc.fit_transform)
    if verbose:
        print(df.head(2))
    X, y = df.drop(["income", "capital-loss"], axis=1).values, df["income"].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X, y = X[~np.isnan(X).any(axis=1)], y[~np.isnan(X).any(axis=1)]
    X = add_noisy_features(X, num_noisy_feats)
    if verbose:
        print(f'Number of samples: {X.shape[0]}, Dimension: {X.shape[1]}')
    return X, y

def load_preprocess_HSLS(verbose=False, num_noisy_feats=0):
    df = pd.read_csv("data/hsls_df.csv")
    df[df <= -7] = np.nan
    df.dropna(inplace=True)
    df["gradebin"] = df["grade9thbin"]
    df["racebin"] = ((df["studentrace"] * 7).astype(int).isin([1, 7])).astype(int)
    df["sexbin"] = df["studentgender"].astype(int)
    df = df.drop(["studentgender", "grade9thbin", "grade12thbin", "studentrace"], axis=1)
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df.rename(columns={"gradebin": "target"}, inplace=True)
    df["racebin"] = df["racebin"] * 2 - 1
    if verbose:
        print(df.head(2))
    y = df["target"].values
    X = df.drop("target", axis=1).values
    X = add_noisy_features(X, num_noisy_feats)
    if verbose:
        print(f'Number of samples: {X.shape[0]}, Dimension: {X.shape[1]}')
    return X, y

def load_preprocess_ENEM(verbose=False,  num_noisy_feats=0):
    df = pd.read_pickle("data/enem-50000-20.pkl")
    df["gradebin"] = df["gradebin"].astype(int)
    df.rename(columns={"gradebin": "target"}, inplace=True)
    df["racebin"] = df["racebin"] * 2 - 1
    if verbose:
        print(df.head(2))
    X, y = df.drop("target", axis=1).values, df["target"].values
    X = X[~np.isnan(X).any(axis=1)]
    X = add_noisy_features(X,  num_noisy_feats)
    if verbose:
        print(f'Number of samples: {X.shape[0]}, Dimension: {X.shape[1]}')
    return X, y


if __name__ == '__main__':

    X_communities, y_comm = load_preprocess_communities(verbose=False, num_noisy_feats=0)
    X_adult, y_adult = load_preprocess_adult(verbose=False, num_noisy_feats=0)
    X_hsls, y_hsls = load_preprocess_HSLS(verbose=False, num_noisy_feats=0)
    X_enem, y_enem = load_preprocess_ENEM(verbose=False, num_noisy_feats=0)

    print(
        'Communities dimensions:', X_communities.shape,
        'Adult dimensions:', X_adult.shape,
        'HSLS dimensions:',X_hsls.shape,
        'ENEM dimensions:', X_enem.shape
        )

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,4, figsize=(10,3))
    axs[0].hist(y_comm)
    axs[0].set_xlabel('Communities')
    axs[1].hist(y_adult)
    axs[1].set_xlabel('Adult')
    axs[2].hist(y_hsls)
    axs[2].set_xlabel('HSLS')
    axs[3].hist(y_enem)
    axs[3].set_xlabel('ENEM')
    fig.suptitle('label distribution per dataset')
    plt.tight_layout()
    plt.savefig('ylabel_hist.png')