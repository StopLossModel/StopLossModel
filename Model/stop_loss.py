from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU, Flatten, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.python.keras.layers.core import Activation
from tensorflow.keras.layers import Input, Concatenate, Reshape


######################### Util Imports #########################
from research_util import root_mean_squared_error, custom_loss, mean_directional_accuracy

######################### Attention Imports #########################
from attention import Attention

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class StopLossPrediction:
    @staticmethod
    def build_trend_detect_branch(inputs, outputs):
        # Encoder - CNN
        x = Conv1D(filters=256, kernel_size=4)(inputs)
        x = Activation(activation="relu")(x)
        x = Conv1D(filters=128, kernel_size=3)(x)
        x = Activation(activation="relu")(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = RepeatVector(outputs)(x)

        # Decoder - LSTM
        x = LSTM(units=320, return_sequences=False)(x)
        x = Activation(activation="relu")(x)
        x = Dense(100)(x)
        x = Activation(activation="relu")(x)
        x = Dense(outputs)(x)
        x = Activation(activation="linear", name="trend_detect")(x)

        return x
    
    @staticmethod
    def build_price_predict_branch(inputs):
        # Encoder - LSTM
        x = LSTM(units=320, return_sequences=True)(inputs)
        x = Activation(activation="relu")(x)
        x = Dropout(0.01)(x)
        x = LSTM(units=320, return_sequences=True)(x)
        x = Activation(activation="relu")(x)
        x = Attention(return_sequences=True)(x)

        # Decoder - CNN
        x = Conv1D(filters=64, kernel_size=2, padding='same')(x)
        x = Activation(activation="relu")(x)
        x = MaxPooling1D(pool_size=2, padding='valid')(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        x = Activation(activation="linear", name="price_predict")(x)

        # Decoder - LSTM
        # x = RepeatVector(7)(x)
        # x = Bidirectional(LSTM(units=100, return_sequences=True))(x)
        # x = Activation(activation="relu")(x)
        # x = Dropout(0.01)(x)
        # x = Dense(100)(x)
        # x = Flatten()(x)
        # x = Dense(1)(x)
        # x = Activation(activation="linear", name="price_predict")(x)

        return x
    
    @staticmethod
    def build_encoder_model(inputs, trend_detect_count):
        trend_detect_branch = StopLossPrediction.build_trend_detect_branch(inputs, trend_detect_count)
        price_pred_branch = StopLossPrediction.build_price_predict_branch(inputs)

        model = Model(inputs=inputs, outputs=[trend_detect_branch, price_pred_branch], name="stop_loss_encoder_model")

        return model

    @staticmethod
    def build(enc_input_shape, dec_input_shape, trend_detect_count):
        enc_inputs = Input(shape=enc_input_shape)
        enc_model = StopLossPrediction.build_encoder_model(enc_inputs, trend_detect_count)

        trend_detect = enc_model.get_layer("trend_detect")
        price_predict = enc_model.get_layer("price_predict")

        # trend_detect_output = Input(trend_detect.output_shape)
        # price_predict_output = Input(price_predict.output_shape)

        dec_input = Input(shape=dec_input_shape)
        dec = Flatten()(dec_input)
        concat = Concatenate(axis=1)([dec, trend_detect.output, price_predict.output])
        concat = RepeatVector(1)(concat)
        x = LSTM(units=320)(concat)
        x = Dense(1)(x)
        x = Activation(activation="linear", name="stop_price_predict")(x)
        dec_model = Model(inputs=[enc_inputs, dec_input], outputs=[x, trend_detect.output, price_predict.output], name="stop_loss_model")

        dec_model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={
                "trend_detect": "mse",
                "price_predict": "mse",
                "stop_price_predict": "mse"
            },
            loss_weights={
                "trend_detect": 1.0,
                "price_predict": 1.0,
                "stop_price_predict": 1.0
            },
            metrics=["accuracy", "mse"],
            run_eagerly=True
        )

        return enc_model, dec_model

class StopLossPrediction2:
    @staticmethod
    def build_encoder_model(price_inputs, trend_inputs, trend_detect_count):
        trend_detect_branch = StopLossPrediction.build_trend_detect_branch(trend_inputs, trend_detect_count)
        price_pred_branch = StopLossPrediction.build_price_predict_branch(price_inputs)

        model = Model(inputs=[trend_inputs, price_inputs], outputs=[trend_detect_branch, price_pred_branch], name="stop_loss_encoder_model")

        return model
    
    @staticmethod
    def build(enc_price_input_shape, enc_trend_input_shape, dec_input_shape, trend_detect_count):
        enc_price_input = Input(shape=enc_price_input_shape)
        enc_trend_input = Input(shape=enc_trend_input_shape)
        enc_model = StopLossPrediction2.build_encoder_model(enc_price_input, enc_trend_input, trend_detect_count)

        trend_detect = enc_model.get_layer("trend_detect")
        price_predict = enc_model.get_layer("price_predict")

        dec_input = Input(shape=dec_input_shape)
        dec = Flatten()(dec_input)
        concat = Concatenate(axis=1)([dec, trend_detect.output, price_predict.output])
        concat = RepeatVector(1)(concat)
        x = LSTM(units=320)(concat)
        x = Dense(1)(x)
        x = Activation(activation="linear", name="stop_price_predict")(x)
        dec_model = Model(inputs=[enc_price_input, enc_trend_input, dec_input], outputs=[x, trend_detect.output, price_predict.output], name="stop_loss_model")

        dec_model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={
                "trend_detect": custom_loss,
                "price_predict": "mse",
                "stop_price_predict": "mse"
            },
            loss_weights={
                "trend_detect": 1.0,
                "price_predict": 1.0,
                "stop_price_predict": 1.0
            },
            metrics=["accuracy", "mse"],
            run_eagerly=True
        )

        return enc_model, dec_model



# if __name__ == '__main__':
#     enc_model, dec_model = StopLossPrediction.build((5000, 9), (5000,1), 5)

#     print(dec_model.summary())