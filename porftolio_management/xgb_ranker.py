import dataset
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score
import pandas as pd
from sklearn.calibration import LabelEncoder
import warnings
from hyperopt import hp, fmin, tpe, Trials, space_eval
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

# Suppress All Warnings
warnings.filterwarnings('ignore')
TOPK=100

def split_dataset(dataset):
    rows_for_test_set = dataset[dataset['gid'] > 110].index
    test_set = dataset.loc[rows_for_test_set].copy()
    dataset.drop(rows_for_test_set, inplace=True)
    
    rows_for_validation_set = dataset[dataset['gid'] % 4 == 0].index
    validation_set = dataset.loc[rows_for_validation_set].copy()  
    dataset.drop(rows_for_validation_set, inplace=True)
    
    return dataset, validation_set, test_set

def encode_target(dataset):
    transformed_data = dataset.copy()
    for gid, group_data in dataset.groupby('gid'):
        targets = group_data['target'].to_numpy()
        percentiles_val = np.percentile(targets, range(0, 101, 20)) 
        star_ratings = []
        for target in group_data['target']:
            rating = 5 - np.searchsorted(percentiles_val[::-1], target)
            star_ratings.append(max(0, rating))
        transformed_data.loc[group_data.index, 'target'] = star_ratings
    return transformed_data['target']

def create_star_ranking(df):
    
    df['rank'] = 0

    for gid in df['gid'].unique():
        gid_group = df[df['gid'] == gid]
        thresholds = np.percentile(gid_group['target'], [25, 50, 75, 100])
        df.loc[(df['gid'] == gid) & (df['target'] > thresholds[3]), 'rank'] = 3
        df.loc[(df['gid'] == gid) & (df['target'] > thresholds[2]) & (df['target'] <= thresholds[2]), 'rank'] = 2
        df.loc[(df['gid'] == gid) & (df['target'] > thresholds[1]) & (df['target'] <= thresholds[1]), 'rank'] = 1
        df.loc[(df['gid'] == gid) & (df['target'] <= thresholds[0]), 'rank'] = 0
    return df

def dcg(scores):
    """Compute the DCG for a list of scores."""
    scores = np.array(scores)
    return np.sum((2**scores - 1) / np.log2(np.arange(2, len(scores) + 2)))

def encode_features(df):
    encoder = LabelEncoder()
    
    df['ticker'] = encoder.fit_transform(df['ticker'])
    df['gid'] = encoder.fit_transform(df['gid'])
    
    return df


def train_ranker(X_train, y_train, train_groups, X_val, y_val, val_groups):
    def objective(params):
        # Convert max_depth and lambdarank_num_leaves to integers
        params['max_depth'] = int(params['max_depth'])
        # params['lambdarank_num_leaves'] = int(params['lambdarank_num_leaves'])
        
        # Train the XGBoost model with the given hyperparameters
        ranker = xgb.XGBRanker(**params, objective='rank:ndcg', random_state=42)
        
        ranker.fit(X_train, y_train, group=train_groups,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_group=[train_groups, val_groups],
            eval_metric=f"ndcg@{TOPK}",
            verbose=False)
            
        y_pred_val = ranker.predict(X_val)
        ndcg = ndcg_score([y_val], [y_pred_val], k=len(y_val))
        
        return -ndcg  # Hyperopt minimizes the objective function

    # Define the hyperparameter search space
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'max_depth': hp.choice('max_depth', np.arange(5, 20, 1,  dtype=int)),
        'gamma': hp.loguniform('gamma', np.log(1e-9), np.log(1.0)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-9), np.log(100.0)),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-9), np.log(100.0)),
        'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
        'n_estimators': hp.choice('n_estimators', np.arange(100, 500, 50, dtype=int))  # Example range, adjust as needed
        # 'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50))
    }
    
    # Run Hyperopt to find the best hyperparameters
    trials = Trials()
    best = fmin(fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials)

    # Retrieve the best parameters
    best_params = space_eval(space, best)
    print("Best hyperparameters:", best_params)
    
    ranker = xgb.XGBRanker(n_jobs=-1, eval_metric=f'ndcg@{TOPK}', random_state=42, objective='rank:ndcg', **best_params)
    ranker.fit(X_train, y_train, group=train_groups,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_group=[train_groups, val_groups])  
    val_ndcg = ranker.evals_result()['validation_1'][f'ndcg@{TOPK}'][-1]
    train_ndcg= ranker.evals_result()['validation_0'][f'ndcg@{TOPK}'][-1]
    return ranker, val_ndcg, train_ndcg

def calculate_ndcg(ranker, X_val, y_val, qid_val):
    y_pred = ranker.predict(X_val)

    unique_qids = np.unique(qid_val)
    ndcg_scores = []
    print(y_pred)
    for qid in unique_qids:
        qid_indices = np.where(qid_val == qid)[0]
        qid_predictions = y_pred[qid_indices]
        qid_targets = y_val[qid_indices]
        
        ndcg = ndcg_score([qid_targets], [qid_predictions], k=len(qid_targets))
        ndcg_scores.append(ndcg)
        print(ndcg)
    
    avg_ndcg = np.mean(ndcg_scores)
    
    return avg_ndcg

def run():
    df, features = dataset.load_and_filter_dataset()
    df = df.dropna(subset=['target'])
    df = create_star_ranking(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    to_remove = ['gid', 'ticker', 'target', 'rank']
    for item in to_remove:
        while item in features:
            features.remove(item)


    df = encode_features(df)
    df_sorted = df.sort_values(by=['gid'])
    train_df, validate_df, test_df = split_dataset(df_sorted)

    train_groups_sizes = train_df.gid.value_counts().sort_index().values.tolist()
    validate_groups_sizes = validate_df.gid.value_counts().sort_index().values.tolist()
    test_groups_sizes = test_df.gid.value_counts().sort_index().values.tolist()

    X_train = train_df[features].values
    y_train = train_df['rank'].values
    # qid_train = train_df['gid'].values

    print(X_train.shape)

    X_val = validate_df[features].values
    y_val = validate_df['rank'].values
    # qid_val = validate_df['gid'].values

    X_test = test_df[features].values
    y_test = test_df['rank'].values
    # qid_test = test_df['gid'].values

    ranker, val_ndcg, train_ndcg = train_ranker(X_train, y_train, train_groups_sizes, X_val, y_val, validate_groups_sizes)
    xgb.plot_tree(ranker, num_trees=3)
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    predictions_val = ranker.predict(X_val[:validate_groups_sizes[1]])
    predictions_train = ranker.predict(X_train[:train_groups_sizes[1]])
    predictions_test = ranker.predict(X_test[:test_groups_sizes[1]])


    print(predictions_val)
    print(predictions_train)
    print(predictions_test)

    print("nDCG score for validation set:", val_ndcg)
    print("nDCG score for train set:", train_ndcg)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_df, validate_df, test_df, predictions_val, predictions_train, predictions_test
    
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, validate_df, test_df, predictions_val, predictions_train, predictions_test = run()
    print(f"y_train{np.unique(y_train)}")
    print(f"y_val{np.unique(y_val)}")
    print(f"y_test{np.unique(y_test)}")
    print(f"predictions_val{np.unique(predictions_val)}")
    print(f"predictions_train{np.unique(predictions_train)}")
    print(f"predictions_test{np.unique(predictions_test)}")