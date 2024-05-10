import pandas as pd

def load_dataset(features, nrows=None):
    print('loading dataset')
    features += ['gid','ticker','target']
    df = pd.read_csv('../data/data.csv', usecols=features, nrows=nrows)
    return df

def load_feature_importance():
    df = pd.read_csv('../data/MI.csv')
    return df

def select_best_features(df, n=2000):
    """
  Selects features based on NaN percentage, score thresholds, and combined score ranking.

  Args:
      df (pd.DataFrame): The DataFrame containing feature scores.
      n (int): The number of features to keep
  Returns:
      pd.DataFrame: The DataFrame containing selected features ranked by combined score.
  """
    # Remove rows with high NaN percentage
    df_filtered = df[df['NaN Percentage'] <= 0.05]  # Keep rows with <= 5% NaN values
    df_filtered = df[df['type'] == 'base']
    df_filtered[['MIR Score', 'MIC Score', 'Pearson r']] = df_filtered[['MIR Score', 'MIC Score', 'Pearson r']].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
    )

    weights = [0.4, 0.35, 0.25]  # Weights for MIR, MIC, Correlation
    df_filtered['Combined Score'] = df_filtered[['MIR Score', 'MIC Score', 'Pearson r']].dot(weights)

    # Sort dataframe by 'Combined Score' and return the best n features
    df_filtered = df_filtered.sort_values(by='Combined Score', ascending=False)
    top_n_features = df_filtered.head(n)
    
    return top_n_features['Feature'].tolist()

def remove_return_columns(df):
    return df.loc[:, ~df.columns.str.contains('return', case=False)]

def remove_trash_features(df, df_best_features):


    test_data = df[df['gid'] == df['gid'].values.tolist()[0]]

    new_feauture_list = []

    for feature in df_best_features:
        unique_valuess = len(list(set(test_data[feature].values.tolist())))
        if unique_valuess > 1:
            new_feauture_list.append(feature)
            if len(new_feauture_list) == 300:
                break
    
    return new_feauture_list


def load_and_filter_dataset(nrows=None):

    feature_importances = load_feature_importance()
    feature_importances = remove_return_columns(feature_importances)

    df_best_features = select_best_features(feature_importances)

    temp_dataset = load_dataset(df_best_features, nrows=1000)

    new_feauture_list = remove_trash_features(temp_dataset, df_best_features)

    dataset = load_dataset(new_feauture_list, nrows)

    return dataset, new_feauture_list

if __name__ == '__main__':
    dataset, features = load_and_filter_dataset()
    print(dataset.head())