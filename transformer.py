import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, layers

from dataset import train_ds, val_ds, test_ds, batch_size
from vectorize import vectorize_layer, vectorize_text, max_features, sequence_length

embedding_dim = 16

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, max_features, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.embedding = layers.Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length)
        self.position_embedding = layers.Embedding(input_dim=sequence_length, output_dim=embedding_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.embedding(inputs)
        embedded_positions = self.position_embedding(positions)
        return embedded_tokens + embedded_positions

class TransformerBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads=2, ff_dim=32, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embedding_dim),
        ])
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=True):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layer_norm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)


model = keras.Sequential([
    TokenAndPositionEmbedding(sequence_length, max_features, embedding_dim),
    TransformerBlock(embedding_dim, num_heads=2, ff_dim=32),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1),
])

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy()])

train_ds2 = train_ds.map(vectorize_text)
val_ds2 = val_ds.map(vectorize_text)
test_ds2 = test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
train_ds2 = train_ds2.cache().prefetch(buffer_size=AUTOTUNE)
val_ds2 = val_ds2.cache().prefetch(buffer_size=AUTOTUNE)
test_ds2 = test_ds2.cache().prefetch(buffer_size=AUTOTUNE)

epochs = 12
history = model.fit(
    train_ds2,
    validation_data=val_ds2,
    epochs=epochs, batch_size=batch_size,
    validation_steps=30)

loss, accuracy = model.evaluate(test_ds2)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
print(model.metrics_names)

# Evaluate the model
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  tf.keras.layers.Activation('sigmoid')
])

print(export_model.predict(np.array([b'This food is good, so I love it'], dtype=object)))  # > 0.5
