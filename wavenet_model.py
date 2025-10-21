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
    dilation_rates = [1, 2, 4, 8, 16] 
    #dilation_rates = [1, 2, 3, 4, 6, 8, 12, 16]  # More granular within same range# Optimal receptive field for sequence length 20
   # dilation_rates = [1, 2, 3, 5, 8, 13]  # Natural growth pattern
    #dilation_rates = [1, 2, 4, 5, 10, 20]  # Matches trading structure
    skip_connections = []
    x = input_tensor
    for dilation in dilation_rates:
        x, skip = wavenet_residual_block(x, nfilt, dilation)
        skip_connections.append(skip)
    return Add()(skip_connections)

###############################################
# 3. Deep WaveNet (used for context data)
###############################################
def deep_wavenet(input, nfilt, wavenet_blocks=4):
    x = input
    for _ in range(wavenet_blocks):
        x = wavenet_block(x, nfilt)
    return x

# --- Classifier Head ---
def build_classification_head(features, n_classes=3, dropout_rate=0.5, l2_reg=0.001, output_bias=None):
    """
    Single direction classification head for López de Prado triple barrier prediction.
    
    Focus on direction signals only - position sizing handled by Bayesian Kelly criterion.
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Global pooling
    pooled = GlobalAveragePooling1D(name='global_pooling')(features)
    
    # Direction classification with regularization
    direction_dropout = Dropout(dropout_rate, name='direction_dropout')(pooled)
    direction_out = Dense(
        n_classes, 
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        bias_initializer=output_bias if output_bias else 'zeros',
        name='direction'
    )(direction_dropout)

    return direction_out


# --- Full Model ---
def build_enhanced_cnn_lstm(
    input_shape=(20, 100),
    n_classes=3,
    wavenet_filters=32,
    wavenet_blocks=4,
    dropout_rate=0.2,
    l2_reg=0.001,
    output_bias=None
):
    """
    WaveNet-based model for López de Prado triple barrier direction prediction.
    
    Single output: direction (DOWN/TIMEOUT/UP)
    Position sizing: Handled externally by Bayesian Kelly criterion
    """
    inputs = Input(shape=input_shape, name="input")
    x = deep_wavenet(inputs, wavenet_filters, wavenet_blocks)
    direction_out = build_classification_head(
        x, n_classes, dropout_rate, l2_reg, output_bias=output_bias
    )
    fin_model = Model(
        inputs=inputs,
        outputs=direction_out  # Single output
    )
    return fin_model


def load_model(model_path):
    """
    Load a saved Keras model 
    
    Args:
        model_path: Path to saved .keras model file
        
    Returns:
        Loaded Keras model
        
    Example:
        >>> from wavenet_model import load_model
        >>> model = load_model('best_model.keras')
    """

    
    return tf.keras.models.load_model(
        model_path, 

        compile=False,  # Compile later with custom losses/metrics if needed

    )
if __name__ == "__main__":
    model = build_enhanced_cnn_lstm(input_shape=(20, 100), n_classes=3, wavenet_filters=32, output_bias=None)

    model.summary()
