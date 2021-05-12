import tensorflow as tf

from encoder import Encoder
from decoder import Decoder


class Model:
    def __init__(self, time_steps, num_layers, units, d_model, num_heads, dropout, name="transformer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = tf.keras.layers.Lambda(Model.create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(inputs)

        look_ahead_mask = tf.keras.layers.Lambda(Model.create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
        dec_padding_mask = tf.keras.layers.Lambda(Model.create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(inputs)

        enc_outputs = Encoder.encoder(time_steps, num_layers, units, d_model, num_heads, dropout)(inputs=[inputs, enc_padding_mask])
        dec_outputs = Decoder.decoder(time_steps, num_layers, units, d_model, num_heads, dropout)([dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=time_steps, use_bias=True, activation='softmax', name="outputs")(dec_outputs)

        self.model = tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name='speedPredictor')

    def train_model(self, x_train, y_train, x_test, y_test, n_epoch, saved_model_path):
        accuracy = []

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # The line that follow comes from : https://www.tensorflow.org/tutorials/keras/save_and_load
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint/cp.ckpt", save_weights_only=True, verbose=1)

        self.model.fit(x_train, y_train, epochs=n_epoch, validation_data=(x_test, y_test), callbacks=[cp_callback])

        self.model.save(saved_model_path, include_optimizer=True)

    @staticmethod
    def create_look_ahead_mask(x):
        seq_len = tf.shape(x)[1]

        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = Model.create_padding_mask(x)

        return tf.maximum(look_ahead_mask, padding_mask)

    @staticmethod
    def create_padding_mask(x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)

        return mask[:, tf.newaxis, tf.newaxis, :]