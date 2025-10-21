import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Multiply, Add, Concatenate, Dense
from tensorflow.keras.models import Model

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
def deep_wavenet(context_input, nfilt):
    x = context_input
    for _ in range(4):
        x = wavenet_block(x, nfilt)
    return x

###############################################
# 4. Build the Generator for Swim Data
###############################################
def build_swim_generator(time_steps=180, nfilt=32):
    latent_in = Input(shape=(time_steps, nfilt), name="latent_input")
    context_in = Input(shape=(time_steps, 8), name="context_input")  # 6 sensor + 1 style + 1 stroke

    # Process latent vector
    latent_out = wavenet_block(latent_in, nfilt)

    # Process context data
    context_processed = deep_wavenet(context_in, nfilt)

    # Branch C: Direct context input
    context_direct = context_in

    # Combine all
    combined = Concatenate(axis=-1)([latent_out, context_processed, context_direct])

    # Outputs: sensor data (6), style (1), stroke (1) = total 8
    output = Dense(8, name="generator_output")(combined)

    model = Model(inputs=[latent_in, context_in], outputs=output, name="Swim_GAN_Generator")
    return model

###############################################
# 5. Build the Discriminator for Swim Data
###############################################
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
    generator_model = build_swim_generator(time_steps=180, nfilt=32)
    discriminator_model = build_swim_discriminator(time_steps=180, nfilt=32)
    generator_model.summary()
    discriminator_model.summary()
    tf.random.set_seed(1337)

    # Conditional generator expects a list of inputs: [noise, style_label, stroke_label]
    test_noise = tf.random.normal([64, 180, 32], mean=0.0, stddev=1.0, seed=1337)  # shape (64, 180, 32)
    #print('randndata = ', tf.keras.backend.eval(test_noise))
    test_sensors = tf.math.tanh(tf.random.normal([64, 180, 6]))

    print("Mean of generated sensor output:", tf.reduce_mean(test_noise).numpy())  # ✅ Now works!
    test_styles = tf.random.uniform([64, 180, 1], minval=0, maxval=6, seed=1337, dtype=tf.float32)  # Random style class indices
    test_stroke = tf.random.uniform([64, 180, 1], minval=0, maxval=2, dtype=tf.float32)  # (64, 180, 1)
    combined_context = tf.concat([test_sensors, test_styles, test_stroke], axis=-1)
    # shape: (batch, time, 13)

    test_output = generator_model([test_noise, combined_context])
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
