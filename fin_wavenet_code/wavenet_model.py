#from tkinter import X
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Multiply, Add, Concatenate, Dense, Activation, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def wavenet_residual_block(input_tensor, nfilt, dilation_rate):
    x = input_tensor
    if x.shape[-1] != nfilt:
        x = Conv1D(filters=nfilt, kernel_size=1, padding='same')(x)

    tanh_out = Conv1D(filters=nfilt, kernel_size=3, dilation_rate=dilation_rate,
                      padding='causal', activation='tanh')(x)
    sigm_out = Conv1D(filters=nfilt, kernel_size=3, dilation_rate=dilation_rate,
                      padding='causal', activation='sigmoid')(x)
    gated = Multiply()([tanh_out, sigm_out])
    skip_out = Conv1D(filters=nfilt, kernel_size=1, padding='same')(gated)
    residual = Conv1D(filters=nfilt, kernel_size=1, padding='same')(gated)
    residual_out = Add()([x, residual])
    return residual_out, skip_out


###############################################
# 2. WaveNet Block
###############################################
def wavenet_block(input_tensor, nfilt):
    dilation_rates = [1, 2, 4, 8, 16, 32, 64, 128]
    skip_connections = []
    x = input_tensor
    for dilation in dilation_rates:
        x, skip = wavenet_residual_block(x, nfilt, dilation)
        skip_connections.append(skip)
    return Add()(skip_connections)

###############################################
# 3. Deep WaveNet (used for context data)
###############################################
def deep_wavenet(input, nfilt):
    x = input
    for _ in range(4):
        x = wavenet_block(x, nfilt)
    return x

# --- Classifier Heads ---
def build_classification_heads(features, num_styles=5, output_bias=None, dropout_rate=0.5, l2_reg=0.001):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
   # x = tf.keras.layers.LayerNormalization(name="stroke_label_layernorm", axis=-1)(features)

    # 2. Volatility forecast (regression) - LOWER dropout, simpler path
    volatility_dropout = Dropout(dropout_rate)(features)  # 0.5x dropout
    volatility_out = Dense(
        1, 
        activation='softplus',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),  # 0.5x L2
        name='volatility'
    )(volatility_dropout)
    
    # 3. Return magnitude (regression) - LOWER dropout, simpler path
    magnitude_dropout = Dropout(dropout_rate)(features)  # 0.5x dropout
    magnitude_out = Dense(
        1, 
        activation='softplus',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),  # 0.5x L2
        name='magnitude'
    )(magnitude_dropout)
    # Global pooling and swim style classification
    pooled = GlobalAveragePooling1D(name='direction_global_pooling')(features)
  #  pooled = Dropout(0.3, name='swim_style_dropout')(pooled)  # 30% dropout

    direction_output = Dense(num_styles, activation='softmax', name='direction_output')(pooled)

    return direction_output, volatility_dropout, magnitude_out

# --- Full Model ---
def build_wavenet_model(input_shape=(20, 100), num_styles=3, nfilt=32, output_bias=None):
    #inputs, features = build_wavenet_backbone(input_shape)
    inputs = Input(shape=input_shape, name="input")
    x = deep_wavenet(inputs, nfilt)
    direction_output, volatility_output, magnitude_output = build_classification_heads(x, num_styles, output_bias=output_bias)
    model = Model(inputs=inputs, outputs=[direction_output, volatility_output, magnitude_output], name="WaveNet_Model_Classifier")

    return model

if __name__ == "__main__":
    model = build_wavenet_model(input_shape=(20, 100), num_styles=3, nfilt=32, output_bias=None)

    model.summary()
