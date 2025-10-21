import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Multiply, Add, Concatenate, Dense, Embedding, Reshape
from tensorflow.keras.models import Model
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.query_conv = Conv1D(self.input_dim, kernel_size=1)
        self.key_conv = Conv1D(self.input_dim, kernel_size=1)
        self.value_conv = Conv1D(self.input_dim, kernel_size=1)
        self.add = Add()
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.input_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        attention_output = tf.matmul(attention_weights, value)
        
        # Residual connection - now both have same shape
        return self.add([attention_output, inputs])
    
    def compute_output_shape(self, input_shape):
        return input_shape

###############################################
# 1. WaveNet Residual Block
###############################################
def wavenet_residual_block(input_tensor, nfilt, dilation_rate):
    x = input_tensor
    if x.shape[-1] != nfilt:
        x = Conv1D(filters=nfilt, kernel_size=1, padding='same')(x)

    tanh_out = Conv1D(filters=nfilt, kernel_size=3, dilation_rate=dilation_rate,
                      padding='causal', activation='tanh')(x)
    sigm_out = Conv1D(filters=nfilt, kernel_size=3, dilation_rate=dilation_rate,
                      padding='causal', activation='sigmoid')(x)
    gated = Multiply()([tanh_out, sigm_out])

    # Add more expressivity with an extra layer
    gated = Conv1D(filters=nfilt, kernel_size=1, padding='same', activation='relu')(gated)
    skip_out = Conv1D(filters=nfilt, kernel_size=1, padding='same')(gated)
    residual = Conv1D(filters=nfilt, kernel_size=1, padding='same')(gated)
    residual_out = Add()([x, residual])
    return residual_out, skip_out

###############################################
# 2. WaveNet Block
###############################################
def wavenet_block(input_tensor, nfilt):
    dilation_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    skip_connections = []
    x = input_tensor
    for dilation in dilation_rates:
        x, skip = wavenet_residual_block(x, nfilt, dilation)
        skip_connections.append(skip)
    return Add()(skip_connections)

###############################################
# 3. Deep WaveNet (used for context data)
###############################################
def deep_wavenet(context_input, nfilt):
    x = context_input
    for _ in range(4):
        x = wavenet_block(x, nfilt)
    return x

###############################################
# 4. Build the Generator for Swim Data
###############################################
def build_swim_generator(time_steps=180, nfilt=32, num_styles=6, seed=1337):
    latent_in = Input(shape=(time_steps, nfilt), name="latent_input")
    swim_style_input = Input(shape=(time_steps, 1), dtype=tf.int32, name="style_label")
    stroke_label_input = Input(shape=(time_steps, 1), dtype=tf.float32, name="stroke_label")

    # Improved embeddings with more capacity
    swim_style_embedding = Embedding(
        input_dim=num_styles + 1,
        output_dim=16,  # Doubled embedding size
        embeddings_constraint=tf.keras.constraints.MaxNorm(1),
        name='gen_swim_style_embed'
    )(swim_style_input)
    swim_style_embedding = Reshape((180, 16), name='gen_swim_reshape')(swim_style_embedding)

    stroke_embedding = Embedding(
        input_dim=3,
        output_dim=8,
        embeddings_constraint=tf.keras.constraints.MaxNorm(1),
        name='gen_stroke_embed'
    )(stroke_label_input)
    stroke_embedding = Reshape((180, 8), name='gen_stroke_reshape')(stroke_embedding)
    
    # Combined context
    context_in = Concatenate(axis=-1)([swim_style_embedding, stroke_embedding])
    
    # Create style-conditional features - this helps the model learn style-specific patterns
    style_condition = Dense(nfilt, activation='tanh')(swim_style_embedding)
    
    # Process latent vector with WaveNet
    latent_out = wavenet_block(latent_in, nfilt)
    
    # Apply style-based modulation - this is key for better style-specific sensor data
    latent_out = Multiply()([latent_out, style_condition])
    
    # Process context with deep WaveNet
    context_processed = deep_wavenet(context_in, nfilt)
    
    # Direct context path for residual information
    context_direct = Dense(nfilt//2, activation='relu')(context_in)
    
    # Combine all information paths
    combined = Concatenate(axis=-1)([latent_out, context_processed, context_direct])
    
    # Additional refinement layers - these help improve sensor fidelity
    x = Conv1D(filters=nfilt*2, kernel_size=3, padding='same', activation='relu')(combined)
    x = Conv1D(filters=nfilt, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=nfilt//2, kernel_size=3, padding='same', activation='relu')(x)

    # Split paths
   # sensor_branch = Dense(6, activation='tanh')(x)
  #  style_branch = Dense(1, activation='linear')(x)  # Or appropriate activation for your style encoding
  #  stroke_branch = Dense(1, activation='sigmoid')(x)  # Or appropriate activation for your stroke encoding

    # Combine for output
  #  output = Concatenate(axis=-1)([sensor_branch, style_branch, stroke_branch])

    # Output layer with tanh activation for bounded values
    output = Dense(8,  name="generator_output")(x)
    
    model = Model(inputs=[latent_in, swim_style_input, stroke_label_input], outputs=output, name="Swim_GAN_Generator")
    return model

###############################################
# 5. Build the Discriminator for Swim Data
###############################################
def build_swim_discriminator_new(time_steps=180, nfilt=32):
    generated_imu = Input(shape=(time_steps, 8), name="generated_imu")
    swim_style_input = Input(shape=(time_steps, 1), name="style_label")  # 1 style
    stroke_label_input = Input(shape=(time_steps, 1), name="stroke_label")  # 1 stroke
    context_in = Concatenate(axis=-1)([swim_style_input, stroke_label_input])  # shape: (batch, time, 24)

    x = Concatenate(axis=-1)([generated_imu, context_in])  # shape = (batch, time, 16)
    x = wavenet_block(x, nfilt)
    x = Concatenate(axis=-1)([x, context_in])
    out = Dense(1, activation='sigmoid', name="real_fake_output")(x)

    model = Model(inputs=[generated_imu, swim_style_input, stroke_label_input], outputs=out, name="Swim_GAN_Discriminator")
    return model

def build_swim_discriminator(time_steps=180, nfilt=32):
    generated_imu = Input(shape=(time_steps, 8), name="generated_imu")
    context_in = Input(shape=(time_steps, 8), name="context_input")  # 6 sensor + 1 style + 1 stroke

    x = Concatenate(axis=-1)([generated_imu, context_in])  # shape = (batch, time, 16)
    x = wavenet_block(x, nfilt)
    x = Concatenate(axis=-1)([x, context_in])
    out = Dense(1, activation='sigmoid', name="real_fake_output")(x)

    model = Model(inputs=[generated_imu, context_in], outputs=out, name="Swim_GAN_Discriminator")
    return model

###############################################
# 6. Instantiate and Summarize the Model
###############################################
if __name__ == "__main__":
    generator_model = build_swim_generator(time_steps=180, nfilt=64, num_styles=6)
    discriminator_model = build_swim_discriminator(time_steps=180, nfilt=32)
    generator_model.summary()
    discriminator_model.summary()
    tf.random.set_seed(1337)

    # Conditional generator expects a list of inputs: [noise, style_label, stroke_label]
    test_noise = tf.random.normal([64, 180, 64], mean=0.0, stddev=1.0, seed=1337)  # shape (64, 180, 32)
    #print('randndata = ', tf.keras.backend.eval(test_noise))
    test_sensors = tf.math.tanh(tf.random.normal([64, 180, 6]))

    print("Mean of generated sensor output:", tf.reduce_mean(test_noise).numpy())  # ✅ Now works!
    test_styles = tf.random.uniform([64, 180, 1], minval=0, maxval=6, seed=1337, dtype=tf.float32)  # Random style class indices
    test_stroke = tf.random.uniform([64, 180, 1], minval=0, maxval=2, dtype=tf.float32)  # (64, 180, 1)
    combined_context = tf.concat([test_sensors, test_styles, test_stroke], axis=-1)
    # shape: (batch, time, 13)

    test_output = generator_model([test_noise, test_styles, test_stroke])
    print(f"\nGenerator test output shape: {test_output.shape}")
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,0]).numpy())  # ✅ Now works!

    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,1]).numpy())  # ✅ Now works!
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,2]).numpy())  # ✅ Now works!
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,3]).numpy())  # ✅ Now works!
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,4]).numpy())  # ✅ Now works!
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,5]).numpy())  # ✅ Now works!
    print(test_output[...,6].numpy())
    # Test discriminator output shapes
    # Discriminator expects sensor data shaped (None, 180, 8)
    disc_output = discriminator_model([test_output, combined_context])
    if isinstance(disc_output, list):
        print("Discriminator outputs:")
        for out in disc_output:
            print(out.shape)
    else:
        print(f"Discriminator single output shape: {disc_output.shape}")
