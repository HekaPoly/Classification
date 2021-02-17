import tensorflow as tf

from multiHeadAttention import MultiHeadAttention
from positionalEncoding import PositionalEncoding


class Encoder:
    @staticmethod
    def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

        attention = MultiHeadAttention(d_model, num_heads, name="attention")\
        ({
            'query': inputs,
            'key': inputs,
            'value': inputs
        })

        attention = tf.keras.layers.Dropout(rate=dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

        outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(inputs=[inputs], outputs=outputs, name=name)

    @staticmethod
    def encoder(time_steps, num_layers, units, d_model, num_heads, dropout, name="encoder"):

        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

        # We implement a linear projection based on Very Deep Self-Attention Networks for End-to-End Speech Recognition
        # Retrieved from https://arxiv.org/abs/1904.13377
        projection = tf.keras.layers.Dense(d_model, use_bias=True, activation='linear')(inputs)

        projection *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        projection = PositionalEncoding(time_steps, d_model)(projection)

        outputs = tf.keras.layers.Dropout(rate=dropout)(projection)

        for i in range(num_layers):
            outputs = Encoder.encoder_layer(units, d_model, num_heads, dropout, "encoder_layer_{}".format(i))([outputs])

        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)