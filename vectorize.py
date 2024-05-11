from tensorflow.keras import layers
import re
import string
import tensorflow as tf

from dataset import train

max_features = 3000
sequence_length = 500


def custom_standardization(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, 'https?://[\w./?&=#-]+', '')
    text = tf.strings.regex_replace(text, r'@\w+', '')
    text = tf.strings.regex_replace(text, r'\\n', ' ')
    text = tf.strings.regex_replace(text, r' +', ' ')
    return tf.strings.regex_replace(text, '[%s]' % re.escape(string.punctuation), '')


vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(train['text'].values)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
