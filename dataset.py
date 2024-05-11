import pandas as pd
import tensorflow as tf


all_data = pd.read_csv('dataset/train.csv', names=['sentiment', 'text'])
all_data = all_data.sample(130000, random_state=0)
# replace sentiment with 0 and 1
all_data['sentiment'] = all_data['sentiment'].replace(1, 0) # Negative
all_data['sentiment'] = all_data['sentiment'].replace(2, 1) # Positive
train = all_data.sample(frac=0.6, random_state=0)
val = all_data.drop(train.index)
test = val.sample(frac=0.5, random_state=0)
val = val.drop(test.index)

batch_size = 64
test_ds = tf.data.Dataset.from_tensor_slices((test['text'].values, test['sentiment'].values)).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((val['text'].values, val['sentiment'].values)).batch(batch_size)
train_ds = tf.data.Dataset.from_tensor_slices((train['text'].values, train['sentiment'].values)).batch(batch_size)
