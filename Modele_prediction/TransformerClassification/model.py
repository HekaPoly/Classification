import tensorflow as tf
import pandas as pd

from encoder import Encoder


class Model:
    def __init__(self, time_steps, num_layers, units, d_model, num_heads, dropout, output_size, name="transformer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

        enc_outputs = Encoder.encoder(time_steps, num_layers, units, d_model, num_heads, dropout)(inputs)

        # We reshape for feeding our FC in the next step
        outputs = tf.reshape(enc_outputs, (-1, time_steps * d_model))

        # We predict our class
        outputs = tf.keras.layers.Dense(units=output_size, use_bias=True, activation='softmax', name="outputs")(outputs)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='audio_class')

    def train_model(self, x_train, y_train, x_test, y_test, n_epoch, saved_model_path):
        accuracy = []

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # The line that follow comes from : https://www.tensorflow.org/tutorials/keras/save_and_load
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint/cp.ckpt", save_weights_only=True, verbose=1)

        history = self.model.fit(x_train, y_train, epochs=n_epoch, validation_data=(x_test, y_test), callbacks=[cp_callback])

        self.model.save(saved_model_path, include_optimizer=True)

        accuracy.append(max(history.history['val_accuracy']))
        accuracy = pd.DataFrame(accuracy, columns=['accuracy'])
