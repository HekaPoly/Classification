import keras.models
import tensorflow as tf

from encoder import Encoder


class Model:
    def __init__(self, time_steps, num_layers, units, d_model, num_heads, dropout, output_size):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

        enc_outputs = Encoder.encoder(time_steps, num_layers, units, d_model, num_heads, dropout)(inputs)

        # We reshape for feeding our FC in the next step
        outputs = tf.reshape(enc_outputs, (-1, time_steps * d_model))

        # We predict our class
        outputs = tf.keras.layers.Dense(units=output_size, activation='linear')(outputs)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='AnglePredictor')
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    def train_model(self, x_train, y_train, x_test, y_test, n_epoch, batch_size):
        self.model.fit(x_train, y_train, epochs=n_epoch, validation_data=(x_test, y_test), batch_size=batch_size, shuffle=True)

    def predict(self, x):
        return self.model.predict(x)

    def load(self, path):
        self.model = keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)
