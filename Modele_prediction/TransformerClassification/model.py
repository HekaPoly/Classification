import tensorflow as tf


from encoder import Encoder


LR_START = 0.001
LR_MAX = 0.06
LR_MIN = 0.0001
LR_RAMPUP_EPOCHS = 45
LR_SUSTAIN_EPOCHS = 2
LR_EXP_DECAY = .9


class Model:
    @staticmethod
    def lrfn(self, epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
        return lr

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
        lr_callback = tf.keras.callbacks.LearningRateScheduler(self.lrfn, verbose=True)

        history = self.model.fit(x_train, y_train, epochs=n_epoch, validation_data=(x_test, y_test), callbacks=[lr_callback])

        self.model.save(saved_model_path, include_optimizer=True)
