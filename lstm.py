import numpy as np
import tensorflow as tf
from tensorflow.keras import losses

from dataset import train_ds, val_ds, test_ds, batch_size
from vectorize import vectorize_layer, vectorize_text

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(vectorize_layer.get_vocabulary()),
        output_dim=64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
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

epochs = 10
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
