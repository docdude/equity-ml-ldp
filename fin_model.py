
# enhanced_model.py
"""
Enhanced CNN-LSTM with Attention and WaveNet components
Based on state-of-the-art financial deep learning architectures
"""
from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K


class AttentionLayer(layers.Layer):
    """
    Bahdanau Attention mechanism for time series
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_u'
        )
        
    def call(self, x):
        # Shape: (batch, time, features)
        score = K.tanh(K.dot(x, self.W) + self.b)
        # score shape: (batch, time, units)
        # self.u shape: (units,)
        # We need to compute attention scores for each time step
        attention_scores = K.sum(score * self.u, axis=-1)
        # attention_scores shape: (batch, time)
        attention_weights = K.softmax(attention_scores, axis=1)
        attention_weights = K.expand_dims(attention_weights, axis=-1)
        
        # Weighted average
        context = K.sum(x * attention_weights, axis=1)
        
        return context, attention_weights


class WaveNetBlock(layers.Layer):
    """
    WaveNet-style dilated causal convolution block
    """
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
    def build(self, input_shape):
        self.conv = layers.Conv1D(
            filters=self.filters * 2,  # For gated activation
            kernel_size=self.kernel_size,
            padding='causal',
            dilation_rate=self.dilation_rate
        )
        self.norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.1)
        
        # 1x1 convolution for residual
        self.conv_res = layers.Conv1D(filters=self.filters, kernel_size=1)
        
        # 1x1 convolution for skip connection
        self.conv_skip = layers.Conv1D(filters=self.filters, kernel_size=1)
        
        # 1x1 convolution for input projection (if dimensions don't match)
        if input_shape[-1] != self.filters:
            self.conv_input = layers.Conv1D(filters=self.filters, kernel_size=1)
        else:
            self.conv_input = None
        
    def call(self, x, training=False):
        # Gated activation unit
        conv_out = self.conv(x)
        conv_out = self.norm(conv_out, training=training)
        
        # Split for gating
        tanh_out = K.tanh(conv_out[..., :self.filters])
        sigm_out = K.sigmoid(conv_out[..., self.filters:])
        
        # Gated activation
        acts = tanh_out * sigm_out
        acts = self.dropout(acts, training=training)
        
        # Residual and skip connections
        # Project input if dimensions don't match
        x_proj = self.conv_input(x) if self.conv_input is not None else x
        res_out = self.conv_res(acts) + x_proj
        skip_out = self.conv_skip(acts)
        
        return res_out, skip_out


def build_enhanced_cnn_lstm(
    input_shape: Tuple[int, int] = (20, 100),  # More features
    n_classes: int = 3,
    
    # WaveNet parameters
    wavenet_filters: int = 32,
    wavenet_blocks: int = 4,
    wavenet_layers_per_block: int = 3,
    
    # CNN parameters  
    conv_filters: List[int] = [64, 128, 256],
    
    # LSTM parameters
    lstm_units: List[int] = [256, 128],
    
    # Attention
    attention_units: int = 128,
    
    # Regularization
    dropout_rate: float = 0.3,
    l2_reg: float = 0.0001,

) -> keras.Model:
    """
    State-of-the-art architecture combining:
    1. WaveNet dilated convolutions for multi-scale patterns
    2. Attention mechanism for focusing on important time steps
    3. CNN for local pattern extraction
    4. LSTM for sequential dependencies
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # Branch 1: WaveNet for multi-scale temporal patterns
    wavenet = inputs
    skip_connections = []
    
    for _ in range(wavenet_blocks):
        for layer in range(wavenet_layers_per_block):
            dilation = 2 ** layer
            wavenet_block = WaveNetBlock(
                filters=wavenet_filters,
                kernel_size=2,
                dilation_rate=dilation
            )
            wavenet, skip = wavenet_block(wavenet)
            skip_connections.append(skip)
    
    # Sum skip connections
    wavenet_out = layers.Add()(skip_connections)
    wavenet_out = layers.GlobalAveragePooling1D()(wavenet_out)
    
    # Branch 2: CNN for local patterns
    cnn = inputs
    for i, filters in enumerate(conv_filters):
        cnn = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(cnn)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.Activation('elu')(cnn)
        
        if i < len(conv_filters) - 1:
            cnn = layers.MaxPooling1D(pool_size=2)(cnn)
        
        cnn = layers.Dropout(dropout_rate)(cnn)
    
    # Branch 3: Bidirectional LSTM with Attention
    lstm = cnn
    for i, units in enumerate(lstm_units):
        # Always return sequences for attention on the last layer
        lstm = layers.Bidirectional(
            layers.LSTM(
                units=units,
                return_sequences=True,  # Always return sequences for attention
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            )
        )(lstm)
        lstm = layers.Dropout(dropout_rate)(lstm)
    
    # Apply attention
    attention_layer = AttentionLayer(attention_units)
    # Get attended features (attention_weights could be used for visualization)
    lstm_attended, _ = attention_layer(lstm)
    
    # Combine all branches
    combined = layers.Concatenate()([wavenet_out, lstm_attended])
    
    # Deep fusion layers
    fusion = layers.Dense(256, activation='elu', kernel_regularizer=keras.regularizers.l2(l2_reg))(combined)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(dropout_rate)(fusion)
    
    fusion = layers.Dense(128, activation='elu', kernel_regularizer=keras.regularizers.l2(l2_reg))(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(dropout_rate)(fusion)
    
    # Multi-task outputs with task-specific regularization
    
    # 1. Direction (classification) - HIGHER dropout for classification
    # Classification needs more regularization than regression
    direction_hidden = layers.Dense(
        64, 
        activation='elu',
        kernel_regularizer=keras.regularizers.l2(l2_reg * 2.0),  # 2x L2 penalty
        name='direction_hidden'
    )(fusion)
    direction_dropout = layers.Dropout(dropout_rate * 1.5)(direction_hidden)  # 1.5x dropout
    # Label smoothing via Dense layer (prevents overconfident predictions)
    direction_out = layers.Dense(
        n_classes, 
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='direction'
    )(direction_dropout)
    
    # 2. Volatility forecast (regression) - LOWER dropout, simpler path
    volatility_dropout = layers.Dropout(dropout_rate * 0.5)(fusion)  # 0.5x dropout
    volatility_out = layers.Dense(
        1, 
        activation='softplus',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=keras.regularizers.l2(l2_reg * 0.5),  # 0.5x L2
        name='volatility'
    )(volatility_dropout)
    
    # 3. Return magnitude (regression) - LOWER dropout, simpler path
    magnitude_dropout = layers.Dropout(dropout_rate * 0.5)(fusion)  # 0.5x dropout
    magnitude_out = layers.Dense(
        1, 
        activation='softplus',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=keras.regularizers.l2(l2_reg * 0.5),  # 0.5x L2
        name='magnitude'
    )(magnitude_dropout)
    
    # Build model
    fin_model = keras.Model(
        inputs=inputs,
        outputs={
            'direction': direction_out,
            'volatility': volatility_out,
            'magnitude': magnitude_out
        }
    )
    

    
    return fin_model


def load_model_with_custom_objects(model_path):
    """
    Load a saved Keras model with custom layers
    
    Args:
        model_path: Path to saved .keras model file
        
    Returns:
        Loaded Keras model
        
    Example:
        >>> from fin_model import load_model_with_custom_objects
        >>> model = load_model_with_custom_objects('best_model.keras')
    """
    custom_objects = {
        'WaveNetBlock': WaveNetBlock,
        'AttentionLayer': AttentionLayer
    }
    
    return tf.keras.models.load_model(
        model_path, 
        custom_objects=custom_objects,
        compile=False,  # Compile later with custom losses/metrics if needed
        safe_mode=False  # Allow loading Lambda layers (safe for our own model)
    )


if __name__ == '__main__':
    """Quick test of model architecture"""
    
    print("🔧 Building financial CNN-LSTM model...")
    model = build_enhanced_cnn_lstm(
        input_shape=(20, 100),
        n_classes=3,
        wavenet_filters=32,
        wavenet_blocks=4,
        wavenet_layers_per_block=3,
        conv_filters=[64, 128, 256],
        lstm_units=[256, 128],
        attention_units=128,
        dropout_rate=0.3,
        l2_reg=0.0001
    )
    
    print("\n📊 Model summary:")
    model.summary()
    
    print("\n✅ Model architecture built successfully!")
    print(f"Total parameters: {model.count_params():,}")
