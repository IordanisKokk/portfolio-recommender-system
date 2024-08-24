###################################################
#Assuming two pandas dataframes 
#train and val, and a list of features
# -- dataframes should be sorted by group_id (gid)
###################################################

from typing import Dict, Tuple
import tensorflow as tf
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import LabelEncoder

from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
from functools import partial
from matplotlib import pyplot as plt

import tensorflow_probability as tfp
from tensorflow_probability import layers as tfpl

import tensorflow_ranking as tfr

print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Probability version: {tfp.__version__}")
print(f"TensorFlow Ranking version: {tfr.__version__}")

import numpy as np

import dataset

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

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
        thresholds = np.percentile(gid_group['target'], [10, 30, 60, 80, 100])
        df.loc[(df['gid'] == gid) & (df['target'] > thresholds[3]), 'rank'] = 4
        df.loc[(df['gid'] == gid) & (df['target'] > thresholds[2]) & (df['target'] <= thresholds[3]), 'rank'] = 3
        df.loc[(df['gid'] == gid) & (df['target'] > thresholds[1]) & (df['target'] <= thresholds[2]), 'rank'] = 2
        df.loc[(df['gid'] == gid) & (df['target'] > thresholds[0]) & (df['target'] <= thresholds[1]), 'rank'] = 1
        df.loc[(df['gid'] == gid) & (df['target'] <= thresholds[0]), 'rank'] = 0
    return df

def encode_features(df):
    encoder = LabelEncoder()
    
    df['ticker'] = encoder.fit_transform(df['ticker'])
    df['gid'] = encoder.fit_transform(df['gid'])
    
    return df

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

X_train = train_df[features].values
y_train = train_df['rank'].values
qid_train = train_df['gid'].values

X_val = validate_df[features].values
y_val = validate_df['rank'].values
qid_val = validate_df['gid'].values

X_test = test_df[features].values
y_test = test_df['rank'].values
qid_test = test_df['gid'].values


#Step 1: scale all the features
sc = StandardScaler()
train_df[features] = sc.fit_transform(train_df[features])
validate_df[features] = sc.transform(validate_df[features])

#Step 2: make data structures for Tensorflow LtR

def _features_and_labels(
    x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
  labels = x.pop("rating")
  return x, labels

def make_dataset(df,features):
    vars = df[['gid']+features].groupby('gid').std().mean(axis=0).sort_values() 
    query_features = vars[vars==0].index
    object_features = [x for x in features if x not in query_features]
    gids =tf.constant(df['gid'].values)
    stocks = tf.constant(df['ticker'].drop_duplicates().sort_values())
    gid_vocabulary = tf.keras.layers.IntegerLookup(
        mask_token=None)
    gid_vocabulary.adapt(gids)
    ticker_vocabulary = tf.keras.layers.IntegerLookup(
        mask_token=None)
    ticker_vocabulary.adapt(stocks)
    tensor_dict = {
        'ticker': tf.constant(df['ticker'].values),
        'gid': tf.constant(df['gid'].values),
        'rating': tf.constant(df['rank'].values,dtype=tf.float32), #this is 4-star rating system based on quantiles
        'object_features': tf.constant(df[object_features].fillna(0).values),
        'query_features': tf.constant(df[query_features].fillna(0).values)
    }
    # Create the dataset
    scores = tf.data.Dataset.from_tensor_slices(tensor_dict)
    key_func = lambda x: gid_vocabulary(x["gid"])
    reduce_func = lambda key, dataset: dataset.batch(10000)
    ds= scores.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=10000)
    ds= ds.map(_features_and_labels)
    ds= ds.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=1))
    return ds,gid_vocabulary,ticker_vocabulary


ds_train,gid_vocabulary_train,ticker_vocabulary_train = make_dataset(train_df,features)
ds_val, gid_vocabulary_val,ticker_vocabulary_val= make_dataset(validate_df,features)

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential([
        tfpl.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)))
    ])
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential([
        tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n)),
        tfpl.MultivariateNormalTriL(n)
    ])
    return posterior_model


class StockRankingModel(tf.keras.Model):
    def __init__(self, num_layers_features, num_layers_query, units_per_layer_feature, units_per_layer_query, dropout_rate, kl_weight):
        super().__init__()
        self.layers_list_feature = []
        self.layers_list_query = []
        
        # Conditional conversion: only if the tensor is ragged
        self.dense_converter_object = tf.keras.layers.Lambda(lambda x: x if isinstance(x, tf.RaggedTensor) else x.to_tensor())
        self.dense_converter_query = tf.keras.layers.Lambda(lambda x: x if isinstance(x, tf.RaggedTensor) else x.to_tensor())

        for index in range(num_layers_features):
            self.layers_list_feature.append(
                tfp.layers.DenseVariational(
                    units_per_layer_feature[index], 
                    make_posterior_fn=posterior, 
                    make_prior_fn=prior, 
                    kl_weight=kl_weight,
                    activation='relu'
                )
            )
            self.layers_list_feature.append(tf.keras.layers.Dropout(dropout_rate))
            
        for index in range(num_layers_query):
            self.layers_list_query.append(
                tfp.layers.DenseVariational(
                    units_per_layer_query[index], 
                    make_posterior_fn=posterior, 
                    make_prior_fn=prior, 
                    kl_weight=kl_weight,
                    activation='relu'
                )
            )
            self.layers_list_query.append(tf.keras.layers.Dropout(dropout_rate))
            
        self.final_transform = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, features):
        object_features = features['object_features']
        query_features = features['query_features']
        
        # Convert ragged tensors to dense if necessary
        if isinstance(object_features, tf.RaggedTensor):
            object_features = object_features.to_tensor()
        if isinstance(query_features, tf.RaggedTensor):
            query_features = query_features.to_tensor()
        
        for layer in self.layers_list_feature:
            object_features = layer(object_features)
        for layer in self.layers_list_query:
            query_features = layer(query_features)

        final_features = tf.keras.layers.Concatenate(axis=-1)([object_features, query_features])
        scores = self.final_transform(final_features)
        scores = tf.squeeze(scores, axis=-1)
        return scores


def create_model(num_layers_features, num_layers_query, units_per_layer_feature, units_per_layer_query, dropout_rate, kl_weight):
    model = StockRankingModel(num_layers_features, num_layers_query, units_per_layer_feature, units_per_layer_query, dropout_rate, kl_weight)
    return model


def train_and_evaluate(params, ds_train, ds_val):
    units_per_layer_query = params['units_per_layer_query']
    units_per_layer_feature = params['units_per_layer_feature']
    num_layers_query = params['num_layers_query']
    num_layers_features = params['num_layers_feature']
    optimizer_class = params['optimizer_class']
    learning_rate = params['learning_rate']
    loss_function = tfr.keras.losses.ApproxNDCGLoss()
    batch_size = params['batch_size']
    dropout_rate = params['dropout_rate']
    kl_weight = params['kl_weight']  # Add this line
    
    # Create the model with kl_weight
    model = create_model(num_layers_features, num_layers_query, units_per_layer_feature, units_per_layer_query, dropout_rate, kl_weight)
    
    eval_metrics = [tfr.keras.metrics.NDCGMetric(name="metric/ndcg@100", topn=100, ragged=True)]
    
    model.compile(optimizer=optimizer_class(learning_rate=learning_rate),       
                  loss=loss_function, metrics=eval_metrics)

    object_features_shape = (None, 240)
    query_features_shape = (None, 60)
    model.build(input_shape={'object_features': object_features_shape, 'query_features': query_features_shape})

    # Add EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(ds_train, validation_data=ds_val, epochs=100, batch_size=batch_size, verbose=1, callbacks=[early_stopping])  
    evaluation = model.evaluate(ds_val, verbose=0)
    score = {"loss": -evaluation[1], "ndcg@100": evaluation[1], "status": STATUS_OK}
    
    return score


partial_function = partial(train_and_evaluate, ds_train=ds_train, ds_val=ds_val)

search_space = {
    'units_per_layer_query': hp.choice('units_per_layer_query', [
        [32, 32, 16, 16],
        [64, 64, 32, 32],
        [128, 128, 64, 64]
    ]),
    'units_per_layer_feature': hp.choice('units_per_layer_feature', [
        [32, 32, 16, 16],
        [64, 64, 32, 32],
        [128, 128, 64, 64]
    ]),
    'activation': hp.choice('activation', ['relu', 'tanh']),
    'optimizer_class': hp.choice('optimizer_class', [
        tf.keras.optimizers.Adagrad,
        tf.keras.optimizers.Adam,
        tf.keras.optimizers.SGD
    ]),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'num_layers_query': hp.choice('num_layers_query', [3, 4]),
    'num_layers_feature': hp.choice('num_layers_feature', [3, 4]),
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
    'kl_weight': hp.uniform('kl_weight', 1e-5, 1e-3)  # Add kl_weight to the search space
}

trials = Trials()

best = fmin(
    fn=partial_function,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)


best_model = space_eval(search_space, best)

print("Best hyperparameters:", best_model)

print(best_model)

def train_best(params, ds_train, ds_val):
    units_per_layer_query = params['units_per_layer_query']
    units_per_layer_feature = params['units_per_layer_feature']
    num_layers_query = params['num_layers_query']
    num_layers_features = params['num_layers_feature']
    optimizer_class = params['optimizer_class']
    learning_rate = params['learning_rate']
    loss_function = tfr.keras.losses.ApproxNDCGLoss(ragged=True)
    batch_size = params['batch_size']
    dropout_rate = params['dropout_rate']
    kl_weight = params['kl_weight']
    
    model = create_model(num_layers_features, num_layers_query, units_per_layer_feature, units_per_layer_query, dropout_rate, kl_weight)
    
    eval_metrics = [tfr.keras.metrics.NDCGMetric(name="metric/ndcg@100", topn=100, ragged=True)]
    
    model.compile(optimizer=optimizer_class(learning_rate=learning_rate),       
                  loss=loss_function, metrics=eval_metrics)

    object_features_shape = (None, 240)
    query_features_shape = (None, 60)
    model.build(input_shape={'object_features': object_features_shape, 'query_features': query_features_shape})

    # Add EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_metric/ndcg@100', start_from_epoch=10, verbose=1, patience=10, restore_best_weights=True)
    
    history = model.fit(ds_train, validation_data=ds_val, epochs=100, batch_size=batch_size, verbose=1, callbacks=[early_stopping])  
    evaluation = model.evaluate(ds_val, verbose=0)
    score = {"loss": -evaluation[1], "ndcg@100": evaluation[1], "status": STATUS_OK}

    print(score)
    return history


model_history = train_best(best_model, ds_train, ds_val)

model_history = model_history.history

# Extract NDCG@100 for training and validation
ndcg_train = model_history['metric/ndcg@100']
ndcg_val = model_history['val_metric/ndcg@100']

# Extract the loss for training and validation
loss_train = model_history['loss']
loss_val = model_history['val_loss']

# Create a new figure
plt.figure(figsize=(12, 6))

# Plot NDCG@100
plt.subplot(1, 2, 1)
plt.plot(ndcg_train, label='Train NDCG@100')
plt.plot(ndcg_val, label='Validation NDCG@100')
plt.xlabel('Epochs')
plt.ylabel('NDCG@100')
plt.title('NDCG@100 Throughout Training')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(loss_train, label='Train Loss')
plt.plot(loss_val, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Throughout Training')
plt.legend()

# Show the plots
plt.show()

print(model_history['val_metric/ndcg@100'][-1])