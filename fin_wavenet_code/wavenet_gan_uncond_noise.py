from tensorflow.keras.layers import (
    Input, Conv1D, Multiply, Add, Concatenate, Dense, Embedding, Reshape,
    Dropout, GaussianNoise
)
from tensorflow.keras.models import Model
import tensorflow as tf

###############################################
# WaveNet Residual Block with Optional Gaussian Noise
###############################################
def wavenet_residual_block(input_tensor, nfilt, dilation_rate, residual_noise_std=None, seed=None):
    x = input_tensor

    if x.shape[-1] != nfilt:
        x = Conv1D(filters=nfilt, kernel_size=1, padding='same')(x)

    # Inject Gaussian noise (if enabled)
    if residual_noise_std:
        x = GaussianNoise(stddev=residual_noise_std, seed=seed)(x)

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
# WaveNet Block (with noise support)
###############################################
def wavenet_block(input_tensor, nfilt, residual_noise_std=None, seed=None):
    dilation_rates = [1, 2, 4, 8, 16, 32, 64, 128]
    skip_connections = []
    x = input_tensor
    for i, dilation in enumerate(dilation_rates):
        # To ensure reproducibility per layer, use a different seed offset
        x, skip = wavenet_residual_block(x, nfilt, dilation, residual_noise_std, seed=seed + i if seed else None)
        skip_connections.append(skip)
    return Add()(skip_connections)

###############################################
# Deep WaveNet (context pathway)
###############################################
def deep_wavenet(context_input, nfilt, residual_noise_std=None, seed=None):
    x = context_input
    for i in range(4):
        x = wavenet_block(x, nfilt, residual_noise_std=residual_noise_std, seed=(seed + 100 + i) if seed else None)
    return x

###############################################
# Swim Data GAN Generator Model
###############################################
def build_swim_generator(
    time_steps=180,
    nfilt=32,
    num_styles=6,
    latent_dropout_rate=0.1,
    latent_noise_std=0.01,
    residual_noise_std=0.01,
    seed=None
):
    # Inputs
    latent_in = Input(shape=(time_steps, nfilt), name="latent_input")
    swim_style_input = Input(shape=(time_steps, 1), dtype=tf.int32, name="style_label")
    stroke_label_input = Input(shape=(time_steps, 1), dtype=tf.float32, name="stroke_label")

    # Latent input augmentation
    x_latent = latent_in
    if latent_dropout_rate:
        x_latent = Dropout(latent_dropout_rate, name="latent_dropout", seed=seed)(x_latent)
    if latent_noise_std:
        x_latent = GaussianNoise(latent_noise_std, name="latent_noise", seed=seed + 1 if seed else None)(x_latent)

    latent_out = wavenet_block(x_latent, nfilt, residual_noise_std=residual_noise_std, seed=seed)

    # Swim style embedding
    swim_style_embedding = Embedding(
        input_dim=num_styles + 1,  # Number of unique swim styles + 1
        output_dim=8,  # Size of the embedding vector
        #embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        embeddings_constraint=tf.keras.constraints.MaxNorm(2, axis=1),
        name='gen_swim_style_embed'
    )(swim_style_input)
    swim_style_embedding = Reshape((time_steps, 8), name='gen_swim_reshape')(swim_style_embedding)
    """
    # Apply dropout for style 0 only
    style_0_mask = tf.cast(tf.equal(swim_style_input, 0), tf.float32)  # shape: (B, T, 1)
    dropout_layer = tf.keras.layers.Dropout(rate=0.5)

    # Apply dropout by multiplying with dropout output
    dropout_noise = dropout_layer(tf.ones_like(swim_style_embedding))
    swim_style_embedding = swim_style_embedding * (1.0 - style_0_mask) + (swim_style_embedding * dropout_noise) * style_0_mask
    """
    # Stroke embedding
    stroke_embedding = Embedding(
            input_dim=3, # Number of unique stroke types + 1
            output_dim=4,  # Increased embedding dimension
           # embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            embeddings_constraint=tf.keras.constraints.MaxNorm(2, axis=1),
            name='gen_stroke_embed'
    )(stroke_label_input)
    stroke_embedding = Reshape((time_steps, 4), name='gen_stroke_reshape')(stroke_embedding)

    # Context concatenation
    context_in = Concatenate(axis=-1)([swim_style_embedding, stroke_embedding])
    context_processed = deep_wavenet(context_in, nfilt, residual_noise_std=residual_noise_std, seed=seed)

    # Final combination
    combined = Concatenate(axis=-1)([latent_out, context_processed, context_in])
    output = Dense(8, use_bias=False, name="generator_output")(combined)

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

def build_swim_discriminator(time_steps=180, nfilt=32, residual_noise_std=None, dropout_rate=None, seed=None):
    generated_imu = Input(shape=(time_steps, 8), name="generated_imu")
    context_in = Input(shape=(time_steps, 8), name="context_input")

    x = Concatenate(axis=-1)([generated_imu, context_in])  # shape = (B, T, 16)
    x = wavenet_block(x, nfilt, residual_noise_std=residual_noise_std, seed=seed)

    # Optional dropout for regularization
    if dropout_rate:
        x = Dropout(dropout_rate, seed=seed)(x)

    # Re-inject context for final classification
    x = Concatenate(axis=-1)([x, context_in])
    out = Dense(1, activation='sigmoid', name="real_fake_output")(x)

    model = Model(inputs=[generated_imu, context_in], outputs=out, name="Swim_GAN_Discriminator")
    return model

###############################################
# 6. Instantiate and Summarize the Model
###############################################
if __name__ == "__main__":
    generator_model = build_swim_generator(
        time_steps=180,
        nfilt=64,
        num_styles=6,
        latent_dropout_rate=0.1,
        latent_noise_std=0.01,
        residual_noise_std=0.01,
        seed=1337
    )

    discriminator_model = build_swim_discriminator(
        time_steps=180,
        nfilt=64,
        residual_noise_std=0.01,
        dropout_rate=0.1,
        seed=1337
    )

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
