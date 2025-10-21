import tensorflow as tf
import numpy as np
import os
import pickle
from wavenet_gan_uncond_v2 import build_swim_discriminator, build_swim_generator
import learning_data
import utils
import datetime
import matplotlib.pyplot as plt
import io
import shutil
from utils_gan import (
    progressive_label_smoothing, smooth_positive_labels, smooth_negative_labels,
    compute_gradient_penalty, soft_categorical_loss, transition_focal_loss, transition_boundary_loss,
    sensor_distribution_loss, sensor_distribution_loss_per_channel, temporal_consistency_loss, temporal_style_loss, transition_sharpness_loss, enhanced_transition_loss, transition_sharpness_loss_debug, spectral_loss, spectral_loss_orig,
    apply_sensor_mean, BalancedAdaptiveLearningRateSchedule,
    compute_batch_class_weights, compute_batch_sample_weights, compute_batch_sample_weights_onehot
)    
from utils_metrics import ( 
    parse_samples, parse_samples_mse,
    compute_sensor_similarity, compute_swim_style_similarity_js, 
    compute_swim_style_similarity_match_mse, compute_swim_style_similarity_match_cos, compute_stroke_similarity, compute_js_divergence_3d_dynamic_bins, compute_cosine_similarity,
    compute_dtw_distance, compute_rmse, compute_rmse_per_swim_style, compute_prd_rmse, compute_sensor_similarity_combined, compute_umap_similarity,
    compute_frechet_distance, compute_mmd
    #plot_samples, plot_swim_styles, plot_to_image
)

# Define paths
data_path = '/teamspace/studios/this_studio/SwimmingModelPython/swim_v2/data_modified_users'
save_path = '/teamspace/studios/this_studio/SwimmingModelPython/swim_v2/tutorial_save_path_gan_wave'
# Define checkpoint directory
checkpoint_dir = "/teamspace/studios/this_studio/SwimmingModelPython/swim_v2/gan_checkpoints_wave"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Ensure save path exists
os.makedirs(save_path, exist_ok=True)

# A list of user names which are loaded
users_all = utils.folders_in_path(data_path)
users = [u for u in users_all]  # Load all users
users.sort(key=int)

# Keeping it simple. Comment this out and use the code above if you want to load everybody
#users = ['2','6','7','11']

# List of users we want to train a model for
users_test = users

# Hyper-parameters for loading data
data_parameters = {
    'users': users,  # Users whose data is loaded
    'labels': [0, 1, 2, 3, 4, 5],  # Labels for swim styles and transitions
    'combine_labels': {0: [0, 5]},  # Combine '0' and '5' as 'transition' for swim style transitions
    'data_columns': ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],  # Sensor data columns
    'stroke_labels': ['stroke_labels'],  # Binary stroke labels: 0 for no stroke, 1 for stroke
    'time_scale_factors': [0.9, 1.1],  # Scale timestamps by 10% faster/slower
    'stroke_range':         None,       # Augments stroke labels in the dataset to include a range around detected peaks
    'win_len': 180,  # Window length in time steps
    'slide_len': 0,  # Slide length for overlapping windows
    'window_normalization': 'tanh_scaled',  # Normalization method for windowed data
    'label_type': 'raw',  # Labeling strategy for overlapping windows
    'majority_thresh': 0.75,  # Threshold for majority labeling
    'validation_set': {
        0: 1,  # Null
        1: 1,  # Freestyle
        2: 1,  # Backstroke
        3: 1,  # Breaststroke
        4: 1,  # Butterfly
        5: 1  # Turn
    },
    # Optional debug keys for easier control and testing:
    'debug': {
        'enable_time_scaling': True,
        'enable_window_normalization': True,
        'use_majority_labeling': True,
    },
}

# Data is loaded and stored in this object
swimming_data = learning_data.LearningData()

# Load recordings from data_path
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=data_parameters['users'],
                        labels=data_parameters['labels'],
                        stroke_labels=data_parameters['stroke_labels'])

# Combine labels if specified
# print("Combining labels...")
# for new_label, old_labels in data_parameters['combine_labels'].items():
#     swimming_data.combine_labels(labels=old_labels, new_label=new_label)
# print("Labels combined.")

# Augment recordings
#swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

# Augments stroke labels in the dataset to include a range around detected peaks
swimming_data.augment_stroke_labels(stroke_range=data_parameters['stroke_range'])

# Compute sliding window locations
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile windows
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])

# GAN training parameters
gan_training_parameters = {
    'lr_generator': 0.0002,  # Reduced from 0.0006
    'lr_discriminator': 0.000005,  # Reduced from 0.000015
    'beta_1': 0.5,
    'batch_size': 64,
    'max_epochs': 400,
    'latent_dim': 100,
    'steps_per_epoch': 100,
    'noise_std': None,
    'mirror_prob': None,
    'random_rot_deg': None,
    'stroke_mask':     False,    # Whether to use a mask for stroke labels
    'stroke_label_output':       True,
    'swim_style_output':         True,
    'output_bias': None
}

# Define GAN components
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']) + 2)  # Include labels + stroke_labels(+ 2) in input

# Users whose data we use for training
users_train = [u for u in users if u != users_test]

# Draw users for each class
train_dict, val_dict = swimming_data.draw_train_val_dicts(users_test, users_per_class=data_parameters['validation_set'])
print("Training dictionary: %s" % train_dict)
print("Validation dictionary: %s" % val_dict)


# Calculate sensor data distribution for training set 
sensor_mean, sensor_std, sensor_bias, channel_mean, channel_bias = utils.calculate_sensor_distribution(
    label_user_dict=train_dict,
    swimming_data=swimming_data,
    data_type="training"
)
# Calculate stroke label distribution for training set (excluding label 0)
training_probabilities, training_mean, training_bias, training_class_weights = utils.calculate_stroke_label_distribution(
    label_user_dict=train_dict,
    swimming_data=swimming_data,
    data_type="training",
    exclude_label=None
)
# Set training bias for generator
gan_training_parameters['output_bias'] = training_bias

# Calculate stroke label distribution for validation set (excluding label 0)
validation_probabilities, validation_mean, validation_bias, validation_class_weights = utils.calculate_stroke_label_distribution(
    label_user_dict=val_dict,
    swimming_data=swimming_data,
    data_type="validation",
    exclude_label=None
)

#generator_parameters = get_default_generator_parameters()
#discriminator_parameters = get_default_discriminator_parameters()

generator = build_swim_generator(
    time_steps=180, 
    nfilt=32,
    num_styles=6
#    generator_parameters=generator_parameters,
  #  use_seed=True,
   # output_bias=gan_training_parameters['output_bias'],
    #sensor_bias=None,
   # channel_bias=channel_bias
)

discriminator = build_swim_discriminator(
    time_steps=180,
    nfilt=32#len(data_parameters['labels']),
 #   discriminator_parameters=discriminator_parameters,
    #use_seed=True,
    #output_bias=gan_training_parameters['output_bias']

)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(
    gan_training_parameters['lr_generator'], beta_1=gan_training_parameters['beta_1']
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    gan_training_parameters['lr_discriminator'], beta_1=gan_training_parameters['beta_1']
)


generator.summary()
discriminator.summary()

# Create an checkpoint tracking variables
epoch_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="epoch_counter")
step_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="step_counter")
log_dir_var = tf.Variable("", dtype=tf.string, trainable=False, name="log_dir")
# Create checkpoint object
checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    gen_optimizer=generator_optimizer,
    disc_optimizer=discriminator_optimizer,
    epoch_counter=epoch_counter,
    step_counter=step_counter,
    log_dir_var=log_dir_var
)

# Restore the latest checkpoint if it exists
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)

if latest_ckpt:
    checkpoint.restore(latest_ckpt)
    print(f"✅ Checkpoint restored from {latest_ckpt}")
    print(f"ℹ️ Resuming optimizer Generator LR: {generator_optimizer.learning_rate.numpy()}, Discriminator LR: {discriminator_optimizer.learning_rate.numpy()}")
    print(f"ℹ️ Resuming training from epoch {epoch_counter.numpy()} and step {step_counter.numpy()}")
    log_dir = log_dir_var.numpy().decode('utf-8')
    print(f"ℹ️ Resuming TensorBoard logging to {log_dir}")
else:
    print("⚠️ No checkpoint found, starting from scratch.")
    epoch_counter.assign(0)
    step_counter.assign(0)
    # Create a new log directory
    log_dir = os.path.join("logs", "gan_wave", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir_var.assign(log_dir)
    print(f"ℹ️ Creating new TensorBoard log directory: {log_dir}")

# TensorBoard setup
#log_dir = os.path.join("logs", "gan_sce", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# Create a summary writer for TensorBoard
summary_writer = tf.summary.create_file_writer(log_dir)

# Get the validation data with weights and mask
x_val, y_val_sparse, y_val_cat, y_stroke_val, val_sample_weights, val_stroke_mask, y_val_raw = swimming_data.get_windows_dict(
    val_dict, return_weights=True, return_mask=True, transition_label=0, return_raw_labels=True
)

y_val_raw_3d = np.expand_dims(y_val_raw, axis=-1)
x_val_combined = np.concatenate((x_val, y_val_raw_3d, y_stroke_val), axis=2)


# The generator used to draw mini-batches
train_gen = swimming_data.batch_generator_dicts(train_dict=train_dict,
                                            batch_size=gan_training_parameters['batch_size'],
                                            noise_std=gan_training_parameters['noise_std'],
                                            mirror_prob=gan_training_parameters['mirror_prob'],
                                            random_rot_deg=gan_training_parameters['random_rot_deg'],
                                            use_4D=False,
                                            swim_style_output=gan_training_parameters['swim_style_output'], 
                                            stroke_label_output=gan_training_parameters['stroke_label_output'],
                                            return_stroke_mask=gan_training_parameters['stroke_mask'])#,
                                         #  return_raw_labels=True)




@tf.function()
def train_step6(real_data, real_styles, real_strokes,
               generator, discriminator,
               gen_optimizer, disc_optimizer,
               batch_size,  style_weights, stroke_weights):


    # Generator lambda factors (precomputed from normalization + importance):
    lambda_adv = 1.0
    lambda_g_real = 50.0
    lambda_style = 1.0
    lambda_stroke = 1.0
   # lambda_spectral = 0.65
   # lambda_temporal_sensor = 0.52
   # lambda_distribution = 10.47
   # lambda_temporal_style = 15

    # Discriminator lambda factors for each sub-loss
    lambda_d_real = 1.0          # Adversarial loss for real samples
    lambda_d_fake = 1.0         # Adversarial loss for fake samples
    lambda_style_real = 1.0      # Style loss for real samples
    lambda_style_fake = 1.0  # Style loss for fake samples (discriminator)
    lambda_stroke_real = 1.0      # Stroke loss for real samples
    lambda_stroke_fake = 1.0 # Stroke loss for fake samples (discriminator)
   # lambda_gradient_penalty = 0.86  # Gradient penalty (regularization)

    # Cast inputs to correct types
    real_data = tf.convert_to_tensor(real_data, tf.float32) # real_data_combined = np.concatenate((real_sensors, real_styles_3d, real_strokes), axis=2, dtype=np.float32)
   # real_context = real_data[..., :8]
   # real_target = real_data[..., :8]

    real_styles = tf.cast(real_styles, tf.float32)
    real_strokes = tf.convert_to_tensor(real_strokes, tf.int32)
    
    # Extract real sensor data (first 6 channels)
    #real_sensors = tf.convert_to_tensor(real_data[..., :6], tf.float32)
  #  real_sensors = real_data[..., :6]
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    mae_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    with tf.GradientTape(persistent=False) as disc_tape, tf.GradientTape(persistent=False) as gen_tape:
        # Generate noise vectors
        noise = tf.random.normal(shape=(batch_size, 180, 32), mean=0.0, stddev=1.0)
        rest_mask = tf.cast(tf.equal(real_styles, 0), tf.float32)
        additional_noise = tf.random.normal(tf.shape(noise)) * 0.5
        noise = noise + rest_mask * additional_noise
        # Generate fake data
        fake_data_gen = generator([noise, real_styles, real_strokes], training=True)
        #fake_data_gen = generator([noise, real_styles, real_strokes], training=True)
       # fake_data_gen = fake_data_disc
        #fake_data_disc, fake_sensors_disc = apply_sensor_mean(fake_data_disc)
        #fake_data_gen, fake_sensors_gen = apply_sensor_mean(fake_data_gen)

        #fake_sensors_disc = fake_data_disc[..., :6]
        #fake_sensors_gen = fake_data_gen[..., :6]
        #fake_styles_gen = fake_data_gen[..., 6]
        # Get discriminator outputs for real and both fake data sets
        real_output = discriminator([real_data, real_styles, real_strokes], training=True)
        fake_output = discriminator([fake_data_gen, real_styles, real_strokes], training=True)

        real_fake_real, style_real, stroke_real = real_output

        real_fake_fake, style_fake, stroke_fake = fake_output
        
        # Progressive label smoothing
        #real_labels = progressive_label_smoothing(epoch, tf.ones_like(real_fake_real))
        #fake_labels = progressive_label_smoothing(epoch, tf.zeros_like(real_fake_fake_disc))
        real_labels = smooth_positive_labels(tf.ones_like(real_fake_real))
        fake_labels = smooth_negative_labels(tf.zeros_like(real_fake_fake))
        #fake_labels = tf.zeros_like(real_fake_fake_disc)

        # Compute discriminator losses using disc fake data
        d_loss_real = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(real_labels, real_fake_real))
        d_loss_fake = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(fake_labels, real_fake_fake))
        
        # Compute generator losses using gen fake data
        g_loss_adv = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.ones_like(real_fake_fake), 
                                              real_fake_fake))

        # Style classification losses
        style_loss_real = tf.reduce_mean(style_weights * 
            mse_loss(real_styles, style_real))
        style_loss_fake = tf.reduce_mean(style_weights * 
            mse_loss(real_styles, style_fake))

        # Stroke detection losses
        stroke_loss_real = tf.reduce_mean(stroke_weights * 
            tf.keras.losses.binary_crossentropy(real_strokes, stroke_real))
        stroke_loss_fake = tf.reduce_mean(stroke_weights * 
            tf.keras.losses.binary_crossentropy(real_strokes, stroke_fake))
        

        g_loss_real = tf.reduce_mean(mse_loss(fake_data_gen, real_data))

        
        # Additional generator losses for sensor quality (using gen fake data)
   #     spectral_loss_term = spectral_loss(real_sensors, fake_sensors_gen)
    #    temporal_sensor_loss_term = temporal_consistency_loss(fake_sensors_gen)
     #   distribution_loss_term = sensor_distribution_loss(real_sensors, fake_sensors_gen)
       # temporal_style_loss_term = transition_sharpness_loss_debug(real_styles, fake_styles_gen)
        #temporal_style_loss_term = transition_focal_loss(real_styles, fake_styles_gen)
    #    temporal_style_loss_term = perfect_transition_loss(real_styles, fake_styles_gen, power=2.0, perfect_weight=100.0)
      #  temporal_style_loss_term = transition_boundary_loss(real_styles, fake_styles_gen, lambda_temporal_style)

        # Compute gradient penalty using disc fake data
       # gradient_penalty = compute_gradient_penalty(real_data, fake_data_disc, discriminator)
        # Compute the total discriminator loss using lambda factors
        total_disc_loss = (
            lambda_d_real            * d_loss_real +
            lambda_d_fake            * d_loss_fake +
            lambda_style_real        * style_loss_real +
            lambda_style_fake   * style_loss_fake +
        
            lambda_stroke_real       * stroke_loss_real +
            lambda_stroke_fake  * stroke_loss_fake 
            #lambda_gradient_penalty  * gradient_penalty
        )



        # Compute the total generator loss using lambda factors
        total_gen_loss = (
            lambda_adv         * g_loss_adv +
            lambda_g_real      * g_loss_real +
            lambda_style       * style_loss_fake +
            lambda_stroke      * stroke_loss_fake
           # lambda_spectral    * spectral_loss_term +
            #lambda_temporal_sensor    * temporal_sensor_loss_term +
            #lambda_distribution * distribution_loss_term +
           # lambda_temporal_style * temporal_style_loss_term
           # temporal_style_loss_term

        )


    # Compute and apply gradients
    disc_grads = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
    gen_grads = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    
    # Clip gradients more aggressively
    disc_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in disc_grads]
    gen_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in gen_grads]
    # Apply gradients
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    return {
        'disc_loss': total_disc_loss,
        'gen_loss': total_gen_loss,
        'disc_real': d_loss_real,
        'disc_fake': d_loss_fake,
        'gen_adv': g_loss_adv,
        'gen_real': g_loss_real,
        'style_real': style_loss_real,
        'style_fake': style_loss_fake,
     #   'style_fake_disc': style_loss_fake_disc,
        'stroke_real': stroke_loss_real,
        'stroke_fake': stroke_loss_fake
       # 'stroke_fake_disc': stroke_loss_fake_disc,
       # 'spectral': spectral_loss_term,
       # 'temporal_sensor': temporal_sensor_loss_term,
       # 'distribution': distribution_loss_term,
       # 'temporal_style': temporal_style_loss_term
        #'gradient_penalty': gradient_penalty
       # 'disc_weights': disc_weights,  # Add discriminator weights
        #'gen_weights': gen_weights  # Add generator weights
    }

def train_gan(generator, discriminator, data_generator, epochs, steps_per_epoch, 
              x_val, y_stroke_val, y_val_raw):
    """Enhanced training loop with monitoring and adaptive features"""
    
    # Initialize metrics history
    history = {
        'disc_loss': [], 'gen_loss': [], 
        'disc_real': [], 'disc_fake': [],
        'gen_adv': [], 'gen_real': [],
        'style_real': [], 'style_fake_gen': [], 'style_fake_disc': [], 
        'stroke_real': [], 'stroke_fake_gen': [], 'stroke_fake_disc': [],
        'spectral': [], 'temporal_sensor': [], 'distribution': [], 'temporal_style': [],
        'gradient_penalty': [] #'disc_weights': [], 'gen_weights': []
    }

    # Initialize the scheduler with starting learning rates:
    balanced_lr_scheduler = BalancedAdaptiveLearningRateSchedule(
        initial_gen_lr=3.25e-04,
        initial_disc_lr=3.00e-05,
        adjustment_factor=1.0,
        tolerance=0.05,
        min_lr=1e-6,
        max_lr=1e-2
    )

    # Get the current epoch from the counter
    start_epoch = epoch_counter.numpy()

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}/{epochs}")
        
        total_d_loss_sum = 0.0
        total_g_loss_sum = 0.0
        step_count = 0
        try:
            epoch_metrics = {key: [] for key in history.keys()}
            
            for step in range(steps_per_epoch):
                step_counter.assign_add(1)
                # Get batch of real data
                real_sensor_data, label_dict = next(data_generator)
                
                # Skip incomplete batches
                if real_sensor_data.shape[0] < gan_training_parameters['batch_size']:
                    continue
                
                # Prepare data
                real_sensors = real_sensor_data #Shape: (64, 180, 6)
                real_styles = label_dict['swim_style_output'] # Shape: (64, 180)
                real_styles_3d = np.expand_dims(real_styles, axis=-1) #S
               # real_styles_onehot = tf.one_hot(real_styles_3d[..., 0], depth=6)  

                real_strokes = label_dict['stroke_label_output']

                real_data_combined = np.concatenate((real_sensors, real_styles_3d, real_strokes), axis=2, dtype=np.float32)

                # Compute sample weights
                style_weights = compute_batch_sample_weights(real_styles_3d, return_weights=False)
                stroke_weights = compute_batch_sample_weights(real_strokes, return_weights=True)
                
                # Training step
                metrics = train_step6(
                    real_data_combined, real_styles_3d, real_strokes,
                    generator, discriminator,
                    generator_optimizer, discriminator_optimizer,
                    gan_training_parameters['batch_size'],
                    style_weights, stroke_weights
                )
                
                # Check for NaN values
                if any(tf.reduce_any(tf.math.is_nan(metrics[m])).numpy() if isinstance(metrics[m], tf.Tensor) else tf.math.is_nan(metrics[m]) for m in metrics):
                    print("NaN values detected, stopping training...")
                    return history

                # Record metrics
                #for key, value in metrics.items():
                 #   epoch_metrics[key].append(value.numpy())
                # Accumulate losses per step
                total_d_loss_sum += metrics['disc_loss']
                total_g_loss_sum += metrics['gen_loss']
                step_count += 1
                # Print progress
                if step % 10 == 0:
                    print(f"Step {step}: D_loss = {metrics['disc_loss']:.4f}, G_loss = {metrics['gen_loss']:.4f}")
                   # print(f"Learning rates - Gen: {generator_optimizer.learning_rate.numpy():.2e}, "
                    #      f"Disc: {discriminator_optimizer.learning_rate.numpy():.2e}")
                step_global = step_counter.numpy()#epoch * steps_per_epoch + step

            # Average epoch metrics
            #for key in history.keys():
             #   history[key].append(np.mean(epoch_metrics[key]))
            # Compute the average losses for the epoch
            avg_d_loss = total_d_loss_sum / step_count
            avg_g_loss = total_g_loss_sum / step_count

            epoch_counter.assign(epoch + 1)
            if epoch % 10 == 0:
                val_fake_data = validate_generator(generator, x_val, y_val_raw, y_stroke_val)                
                save_models(generator, discriminator, epoch, save_path)

                # **Adaptive Learning Rate Adjustment**
                #avg_d_loss = history['disc_loss'][-1]  # Get the last epoch's average D_loss
                #avg_g_loss = history['gen_loss'][-1]  # Get the last epoch's average G_loss

                new_gen_lr, new_disc_lr = balanced_lr_scheduler(avg_d_loss, avg_g_loss)
                generator_optimizer.learning_rate.assign(new_gen_lr)
                discriminator_optimizer.learning_rate.assign(new_disc_lr)     
                checkpoint.save(file_prefix=checkpoint_prefix)
                print(f"✅ Checkpoint saved at epoch {epoch} (Next start: {epoch + 1})")
            # TensorBoard logging
            with summary_writer.as_default():
               #for key, value in history.items():
                #    tf.summary.scalar(f'{key}/epoch_average', value[-1], step=epoch)
                
                # Log all metrics from train_step
               # for key, value in metrics.items():
                #    tf.summary.scalar(f'step/{key}', value, step=step_global)

                for key, value in metrics.items():
                    if isinstance(value, tf.Tensor) and value.shape.rank > 0:  # If it's an array
                        # Option 1: Log each component
                        for i, v in enumerate(value):
                            tf.summary.scalar(f'step/{key}_{i}', v.numpy(), step=step_global)
                    else:
                        # Log scalar values directly
                        tf.summary.scalar(f'step/{key}', value.numpy() if isinstance(value, tf.Tensor) else value, step=step_global)
                # Log samples and counts

                #if epoch % 50 == 0:
                 #   tf.summary.histogram('real_samples', x_val, step=step_global)
                  #  tf.summary.histogram('fake_samples', val_fake_data, step=step_global)

                
                # Log sample visualizations
                if epoch % 10 == 0:
                    real_samples = x_val
                    real_strokes, fake_strokes, real_styles, fake_styles, real_sensors, fake_sensors = parse_samples_mse(real_samples, val_fake_data)
                    real_stroke_count = np.sum(real_strokes, dtype=np.int32)
                    fake_strokes = np.where(fake_strokes > 0.5, 1, 0)

                    fake_stroke_count = np.sum(fake_strokes)
                    print(f"\nReal Stroke Count: {real_stroke_count} | Fake Stroke Count: {fake_stroke_count}")
                    #print("Sum of each row (should be ~1):", np.sum(fake_styles, axis=-1))


                    tf.summary.scalar('real_stroke_count', real_stroke_count, step=step_global)
                    tf.summary.scalar('fake_stroke_count', fake_stroke_count, step=step_global)

                    # Compute similarities
                    stroke_similarity = compute_stroke_similarity(real_strokes, fake_strokes)
                    #swim_style_similarity_js = compute_swim_style_similarity_js(real_styles, fake_styles)
                    swim_style_similarity_match = compute_swim_style_similarity_match_mse(real_styles, fake_styles)
                #    swim_style_similarity_match_cos = compute_swim_style_similarity_match_cos(real_styles, fake_styles)

                    sensor_similarity = compute_sensor_similarity(real_sensors, fake_sensors)
                    real_variance = np.var(real_samples, axis=0)
                    print(f"   Real sample variance (mean across features): {np.mean(real_variance):.4f}")

                    fake_variance = tf.math.reduce_variance(val_fake_data, axis=0).numpy()
                    print(f"   Fake sample variance (mean across features): {np.mean(fake_variance):.4f}")

                  #  sensor_similarity_js = compute_js_divergence_3d_dynamic_bins(real_sensors, fake_sensors)
                    #sensor_similarity_combined, umap_sim, rmse_score, dtw_score, cosine_sim = compute_sensor_similarity_combined(real_sensors, fake_sensors)
                  #  sensor_rmse = compute_rmse(real_sensors, fake_sensors)
                    sensor_rmse_swim_style = compute_rmse_per_swim_style(real_sensors, fake_sensors, real_styles)
                    sensor_prd_rmse_score = compute_prd_rmse(real_sensors, fake_sensors)

                    #sensor_dtw = compute_dtw_distance(real_sensors, fake_sensors)

                    sensor_fd_score = compute_frechet_distance(real_sensors, fake_sensors)
                    #sensor_mmd_score = compute_mmd(real_sensors, fake_sensors, kernel='rbf', sigma=1.0)

                    # Compute overall similarity (weighted sum)
                    #overall_similarity = 0.4 * stroke_similarity + 0.3 * swim_style_similarity_match + 0.3 * sensor_similarity

                    # Log to TensorBoard
                    tf.summary.scalar("Similarity/Stroke", stroke_similarity, step=epoch)
                    #tf.summary.scalar("Similarity/Swim_StyleJS", swim_style_similarity_js, step=epoch)
                    tf.summary.scalar("Similarity/Swim_StyleMatch", swim_style_similarity_match, step=epoch)
             #       tf.summary.scalar("Similarity/Swim_StyleMatch_Cos", swim_style_similarity_match_cos, step=epoch)

                    tf.summary.scalar("Similarity/Sensors", sensor_similarity, step=epoch)
                  #  tf.summary.scalar("Similarity/SensorsJs", sensor_similarity_js, step=epoch)
                   # tf.summary.scalar("Similarity/Sensors_RMSE", sensor_rmse, step=epoch)
                    tf.summary.scalar("Similarity/Sensors_RMSE_Swim", sensor_rmse_swim_style, step=epoch)
                    tf.summary.scalar("Similarity/Sensors_RMSE_PRD", sensor_prd_rmse_score, step=epoch)

                    #tf.summary.scalar("Similarity/Sensors_DTW", sensor_dtw, step=epoch)

                    tf.summary.scalar("Similarity/Swim_Style FD", sensor_fd_score, step=epoch)
                   # tf.summary.scalar("Similarity/Swim_Style MMD", sensor_mmd_score, step=epoch)

                    #tf.summary.scalar("Similarity/Overall", overall_similarity, step=epoch)

                    print(f"Epoch {epoch}: Stroke Similarity={stroke_similarity:.2f}% | "
                        #f"Swim Style SimilarityJS={swim_style_similarity_js:.2f}% | "
                        f"Swim Style SimilarityMatch={swim_style_similarity_match:.2f}% | "
                   #     f"Swim Style SimilarityMatchCos={swim_style_similarity_match_cos:.2f}% | "

                        f"Sensor Similarity={sensor_similarity:.2f}% | "
                     #   f"Sensor SimilarityJs={sensor_similarity_js:.2f}% | "
                        #f"Sensor Similarity Combined\={sensor_similarity_combined:.2f}% | "
                        #f"Sensor UMAP={umap_sim:.2f}% | "
                      #  f"Sensor RMSE={sensor_rmse:.2f} | "
                        f"Sensor RMSE_Swim={sensor_rmse_swim_style:.2f} | "
                        f"PRD RMSE Score={sensor_prd_rmse_score:.2f}% | "

                        #f"Sensor DTW={sensor_dtw:.2f} | "

                        f"Sensor Fréchet Distance={sensor_fd_score:.4f} | ")
                      #  f"Sensor Maximum Mean Discrepancy={sensor_mmd_score:.4f}")
                        #f"Overall Similarity={overall_similarity:.2f}%")
                    summary_writer.flush()

                    start_index = x_val.shape[0] // 2
                    num_samples = 5
                    fig = plot_samples(
                        real_samples[start_index:start_index + num_samples],
                        val_fake_data[start_index:start_index + num_samples].numpy(),
                        num_samples=5
                    )
                    tf.summary.image("Sample Comparison", plot_to_image(fig), step=epoch)
                    del fig


                    fig_rest = plot_continuous_style_samples(
                        real_samples,
                        val_fake_data.numpy(),
                        target_style=0,  # 0 = Rest
                        channels=[3, 4, 5],  # Gyroscope channels
                        max_samples=1000,
                        add_markers=False  # Add vertical lines at sequence boundaries
                    )

                    tf.summary.image("Rest Sample Comparison", plot_to_image(fig_rest), step=epoch)
                    if fig_rest is not None:
                        del fig_rest

                    # Plot swim styles
                    fig_swim_style = plot_swim_styles(real_styles, fake_styles)
                    tf.summary.image("Swim_style Comparison", plot_to_image(fig_swim_style), step=epoch)
                    del fig_swim_style
                    tf.keras.backend.clear_session()


                # Log learning rates
                tf.summary.scalar('learning_rate/generator', 
                                generator_optimizer.learning_rate.numpy(), 
                                step=epoch)
                tf.summary.scalar('learning_rate/discriminator', 
                                discriminator_optimizer.learning_rate.numpy(), 
                                step=epoch)
              #  if epoch > stability_window:
               #     tf.summary.scalar('training/unstable_epochs', 
                #                    float(unstable_epochs), 
                 #                   step=epoch)
                tf.summary.flush()
        
        
        except tf.errors.InvalidArgumentError as e:
            print(f"Error occurred: {e}")
            print("Stopping training due to numerical instability")
            return history
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            return history
    
    return history


def validate_generator(generator, x_val, y_val_raw, y_stroke_val):
    """Validation and visualization function"""
    # Generate validation samples
   # val_noise = tf.random.normal([len(x_val), 180, 32])    
    val_noise = tf.random.normal(shape=(len(x_val), 180, 32), mean=0.0, stddev=1.0)
    y_val_raw_3d = np.expand_dims(y_val_raw, axis=-1)
    val_styles = tf.convert_to_tensor(y_val_raw_3d, dtype=tf.float32)          
    val_strokes = tf.convert_to_tensor(y_stroke_val, dtype=tf.int32)          

    val_fake_data = generator([val_noise, val_styles, val_strokes], training=False)
     
    # Compare real vs fake data shapes (should both be [batch, 180, 8])
    print(f"\nValidation Stats:")
    print(f"   Real data shape: {x_val.shape} (sensors + labels)")
    print(f"   Generated shape: {val_fake_data.shape}")
 
    return val_fake_data

def save_models(generator, discriminator, epoch, save_path):
    """Save generator and discriminator models"""
    # Create model save directories if they don't exist
    model_save_path = os.path.join(save_path, "models")
    gen_save_dir = os.path.join(model_save_path, "generator")
    disc_save_dir = os.path.join(model_save_path, "discriminator")
    
    os.makedirs(gen_save_dir, exist_ok=True)
    os.makedirs(disc_save_dir, exist_ok=True)
    
    # Save models
    generator.save(os.path.join(gen_save_dir, f'generator_epoch_{epoch}.keras'))
    discriminator.save(os.path.join(disc_save_dir, f'discriminator_epoch_{epoch}.keras'))
    
    # Save weights separately (as backup)
    generator.save_weights(os.path.join(gen_save_dir, f'generator_weights_epoch_{epoch}.weights.h5'))
    discriminator.save_weights(os.path.join(disc_save_dir, f'discriminator_weights_epoch_{epoch}.weights.h5'))
    
    # Save training parameters
    params = {
        'epoch': epoch,
        'gan_training_parameters': gan_training_parameters
     #   'generator_parameters': generator_parameters,
      #  'discriminator_parameters': discriminator_parameters
    }
    
    with open(os.path.join(model_save_path, f'training_params_epoch_{epoch}.pkl'), 'wb') as f:
        pickle.dump(params, f)


def plot_to_image(fig):
    """Converts a Matplotlib figure to a PNG image tensor and releases memory."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')  # Save figure
    buf.seek(0)

    # Convert PNG buffer to a TF image tensor
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.expand_dims(image, 0)  # Add batch dimension

    # **Close figure and free memory**
    plt.close(fig)
    buf.close()

    return image

def plot_samples(real_samples, fake_samples, num_samples=5):
    """Plot real and generated swim stroke samples."""
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 10), facecolor='white')
    colors = plt.cm.viridis(np.linspace(0, 1, 6))

    for i in range(num_samples):
        # Real Sample
        ax_real = axes[0, i]
        for ch in range(3):
            sensor_data = real_samples[i, :, ch]
            ax_real.plot(sensor_data, color=colors[ch], alpha=0.7, label=f'Sensor {ch+1}')
            if ch == 0:
                stroke_indices = np.where(real_samples[i, :, 7] > 0.5)[0]
                ax_real.scatter(stroke_indices, sensor_data[stroke_indices], color='red', marker='o', s=10,
                                edgecolor='black', zorder=5, label='Stroke Peaks (Real)')
        ax_real.set_title(f'Real Sample {i+1}')
        ax_real.legend(loc='upper right')

        # Fake Sample
        ax_fake = axes[1, i]
        for ch in range(3):
            sensor_data = fake_samples[i, :, ch]
            ax_fake.plot(sensor_data, color=colors[ch], alpha=0.7)
            if ch == 0:
                fake_strokes = np.where(fake_samples[i, :, 7] > 0.5)[0]
                ax_fake.scatter(fake_strokes, sensor_data[fake_strokes], color='blue', marker='o', s=10,
                                edgecolor='black', zorder=5, label='Stroke Peaks (Generated)')
        ax_fake.set_title(f'Generated Sample {i+1}')
        ax_fake.legend(loc='upper right')

    plt.tight_layout()
    return fig  # Return the figure for TensorBoard logging

def plot_swim_styles(real_styles, fake_styles):
    """
    Plots real vs. fake swim styles as two separate continuous graphs stacked vertically.

    Args:
        real_samples (numpy array): Real swim style predictions (batch_size, 180, 13).
        fake_samples (numpy array): Generated swim style predictions (batch_size, 180, 13).
    """
    num_samples = real_styles.shape[0]  # Number of samples
    time_steps = real_styles.shape[1]   # 180 time steps
    total_steps = num_samples * time_steps  # Total points to plot

    # Extract swim style predictions and convert to categorical (argmax across 6 style probabilities)
    real_styles = real_styles.flatten() #np.argmax(real_samples[:, :, 6], axis=-1).flatten()  # Shape: (num_samples * 180,)
    fake_styles = np.round(fake_styles.flatten())  # Shape: (num_samples * 180,)

    # Scale x-axis to make visualization manageable
    x_values = np.arange(total_steps) # Normalize index for better visualization

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

    # Real Swim Styles (Top Plot)
    axes[0].plot(x_values, real_styles, linestyle='-', color='blue', alpha=0.7)
    axes[0].set_ylabel("Real Swim Style")
    axes[0].set_yticks(range(6))  # Swim styles (0-5)
    axes[0].set_title("Real Swim Styles")

    # Fake Swim Styles (Bottom Plot)
    axes[1].plot(x_values, fake_styles, linestyle='-', color='red', alpha=0.7)
    axes[1].set_ylabel("Fake Swim Style")
    axes[1].set_yticks(range(6))  # Swim styles (0-5)
    axes[1].set_xlabel("Sample Index (scaled)")
    axes[1].set_title("Fake Swim Styles")

    plt.tight_layout()
    return plt.gcf()


def plot_continuous_style_samples(real_samples, fake_samples, target_style=0, channels=None, max_samples=1000, add_markers=True):
    """
    Plot real and generated samples for a specific style continuously in two stacked graphs.
    
    Args:
        real_samples: Real data samples with shape [batch_size, 180, 8]
        fake_samples: Fake data samples with shape [batch_size, 180, 8]
        target_style: Style to extract (default: 0 for rest)
        channels: List of sensor channels to plot (default: [3, 4, 5])
        max_samples: Maximum number of samples to process
        add_markers: Whether to add markers at sequence boundaries
        
    Returns:
        matplotlib figure
    """
    # Limit the number of samples to process
    real_samples = real_samples[:max_samples]
    fake_samples = fake_samples[:max_samples]
    
    # Default to gyroscope channels if not specified
    if channels is None:
        channels = [3, 4, 5]  # Default to gyroscope channels
    
    # Extract sensor data and styles
    real_sensors = real_samples[..., :6]  # First 6 channels are sensors
    fake_sensors = fake_samples[..., :6]

    real_styles = real_samples[..., 6]  # (batch_size, 180)
    fake_styles = np.abs(np.round(fake_samples[..., 6]))  # (batch_size, 180)

    # Create masks for target style timesteps
    real_style_mask = (real_styles == target_style)  # Shape: [batch_size, 180]
    fake_style_mask = (fake_styles == target_style)  # Shape: [batch_size, 180]
    
    # Initialize lists to store style sensor data and sequence boundaries
    real_style_data = []
    fake_style_data = []
    real_boundaries = [0]  # Start with 0
    fake_boundaries = [0]  # Start with 0
    
    # Process each sequence
    for i in range(real_sensors.shape[0]):
        # Get indices where style matches target in this sequence
        style_indices = np.where(real_style_mask[i])[0]
        
        # Extract sensor values at those indices
        if len(style_indices) > 0:
            real_style_data.append(real_sensors[i, style_indices])
            real_boundaries.append(real_boundaries[-1] + len(style_indices))
    
    for i in range(fake_sensors.shape[0]):
        # Get indices where style matches target in this sequence
        style_indices = np.where(fake_style_mask[i])[0]
        
        # Extract sensor values at those indices
        if len(style_indices) > 0:
            fake_style_data.append(fake_sensors[i, style_indices])
            fake_boundaries.append(fake_boundaries[-1] + len(style_indices))

    # Ensure we have at least one valid style sample
    if len(real_style_data) == 0 or len(fake_style_data) == 0:
        print(f"⚠️ No valid style {target_style} samples found in real or generated data.")
        return None

    # Flatten and concatenate all samples for continuous plotting
    real_continuous = np.concatenate(real_style_data, axis=0)  # Shape: (total_timesteps, 6)
    fake_continuous = np.concatenate(fake_style_data, axis=0)  # Shape: (total_timesteps, 6)

    # Print statistics
    print(f"Style {target_style} statistics:")
    print(f"  Real data: {real_continuous.shape[0]} timesteps from {len(real_style_data)} sequences")
    print(f"  Fake data: {fake_continuous.shape[0]} timesteps from {len(fake_style_data)} sequences")
    print(f"  Avg. timesteps per sequence: Real={real_continuous.shape[0]/len(real_style_data):.1f}, "
          f"Fake={fake_continuous.shape[0]/len(fake_style_data):.1f}")

    # Determine number of total time steps
    time_steps_real = np.arange(real_continuous.shape[0])  # X-axis time steps for real
    time_steps_fake = np.arange(fake_continuous.shape[0])  # X-axis time steps for fake

    # Create two stacked subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), facecolor='white')
    colors = plt.cm.viridis(np.linspace(0, 1, 6))  # Generate distinct colors

    # Style names for title
    style_names = {
        0: "Rest",
        1: "Freestyle",
        2: "Breast",
        3: "Back",
        4: "Butterfly",
        5: "Turn"
    }
    style_name = style_names.get(target_style, f"Style {target_style}")
    
    # Channel names
    channel_names = {
        0: "ACC_X",
        1: "ACC_Y",
        2: "ACC_Z",
        3: "GYRO_X",
        4: "GYRO_Y",
        5: "GYRO_Z"
    }
    # Find global min and max across both real and fake data
    for ch in channels:
        all_data = np.concatenate([real_continuous[:, ch], fake_continuous[:, ch]])
        global_min = np.min(all_data)
        global_max = np.max(all_data)
        padding = (global_max - global_min) * 0.1

        # Apply to both axes
        axes[0].set_ylim(global_min - padding, global_max + padding)
        axes[1].set_ylim(global_min - padding, global_max + padding)

    # Plot real sensor data (Top)
    for ch in channels:
        axes[0].plot(time_steps_real, real_continuous[:, ch], color=colors[ch], 
                   alpha=0.8, linestyle='-', label=f'{channel_names.get(ch, f"Sensor {ch+1}")}')
    
    # Add sequence boundary markers if requested
    if add_markers and len(real_boundaries) > 2:
        for boundary in real_boundaries[1:-1]:
            axes[0].axvline(x=boundary, color='r', linestyle='--', alpha=0.3)
    
    axes[0].set_title(f"Real {style_name} Style Sensor Data")
    axes[0].set_ylabel("Sensor Value")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # Plot fake sensor data (Bottom)
    for ch in channels:
        axes[1].plot(time_steps_fake, fake_continuous[:, ch], color=colors[ch], 
                   alpha=0.8, linestyle='-', label=f'{channel_names.get(ch, f"Sensor {ch+1}")}')
    
    # Add sequence boundary markers if requested
    if add_markers and len(fake_boundaries) > 2:
        for boundary in fake_boundaries[1:-1]:
            axes[1].axvline(x=boundary, color='r', linestyle='--', alpha=0.3)
    
    axes[1].set_title(f"Generated {style_name} Style Sensor Data")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Sensor Value")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    plt.tight_layout()
    return fig

def convert_soft_labels_to_hard(swim_styles, num_styles):
    """
    Converts soft one-hot labels into hard labels by sampling from their probability distributions.

    Args:
        real_styles (numpy array): Shape (num_samples, 180, num_styles) - Soft one-hot labels.
        num_styles (int): Number of swim styles.

    Returns:
        numpy array: Shape (num_samples * 180,) - Hard labels sampled from soft distributions.
    """
    # Ensure probabilities sum to exactly 1
    #normalized_probs = np.round(swim_styles / np.sum(swim_styles, axis=-1, keepdims=True), decimals=8)

    # Ensure no row has NaN or invalid sum due to precision errors
    #normalized_probs = np.nan_to_num(normalized_probs, nan=0, posinf=0, neginf=0)

    # Step 1: Renormalize using np.maximum to prevent division by near-zero values
    swim_styles /= np.maximum(np.sum(swim_styles, axis=-1, keepdims=True), 1e-8)

    # Step 2: Explicit rounding to force exact sums
    swim_styles = np.round(swim_styles, decimals=8)

    # Final verification
    #print("Sum of each row (should be exactly 1):", np.sum(swim_styles, axis=-1))

    return np.array([
        np.random.choice(num_styles, p=soft_row)
        for soft_row in swim_styles.reshape(-1, num_styles)
    ])

def plot_swim_styles_soft(real_styles, fake_styles):
    """
    Plots real vs. fake swim styles as two separate continuous line graphs.
    
    Instead of using `argmax`, this function samples swim styles from the soft one-hot distributions
    using `np.random.choice`, which respects the probabilistic nature of the labels.

    Args:
        real_styles (numpy array): Real swim style distributions (batch_size, 180, num_styles).
        fake_styles (numpy array): Generated swim style distributions (batch_size, 180, num_styles).
    """
    num_samples, time_steps, num_styles = real_styles.shape  # Shape: (batch_size, 180, num_styles)

    # Convert soft one-hot labels into discrete swim styles by sampling
    row_real_styles = convert_soft_labels_to_hard(real_styles, num_styles)#np.array([np.random.choice(num_styles, p=soft_row) for soft_row in real_styles.reshape(-1, num_styles)])
    row_fake_styles = convert_soft_labels_to_hard(fake_styles, num_styles)#np.array([np.random.choice(num_styles, p=soft_row) for soft_row in fake_styles.reshape(-1, num_styles)])

    # X-axis values (scaled by total samples)
    x_values = np.arange(len(row_real_styles))

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

    # Real Swim Styles (Top Plot)
    axes[0].plot(x_values, row_real_styles, linestyle='-', color='blue', alpha=0.7)
    axes[0].set_ylabel("Real Swim Style")
    axes[0].set_yticks(range(num_styles))  # Swim styles (0-5)
    axes[0].set_title("Real Swim Styles")

    # Fake Swim Styles (Bottom Plot)
    axes[1].plot(x_values, row_fake_styles, linestyle='-', color='red', alpha=0.7)
    axes[1].set_ylabel("Fake Swim Style")
    axes[1].set_yticks(range(num_styles))  # Swim styles (0-5)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_title("Fake Swim Styles")

    plt.tight_layout()
    return plt.gcf()

def check_training_stability(gen_loss_history, disc_loss_history, window=50):
    """Check if training is stable based on loss history"""
    if len(gen_loss_history) < window:
        return True
    
    recent_gen_loss = gen_loss_history[-window:]
    recent_disc_loss = disc_loss_history[-window:]
    
    gen_std = np.std(recent_gen_loss)
    disc_std = np.std(recent_disc_loss)
    
    gen_trend = np.mean(recent_gen_loss[-10:]) - np.mean(recent_gen_loss[:10])
    disc_trend = np.mean(recent_disc_loss[-10:]) - np.mean(recent_disc_loss[:10])
    
    # Check if losses are stable and not diverging
    is_stable = (gen_std < 1.0 and 
                disc_std < 1.0 and 
                abs(gen_trend) < 0.5 and 
                abs(disc_trend) < 0.5)
    
    return is_stable

history = train_gan(
    generator, 
    discriminator, 
    data_generator=train_gen, 
    epochs=gan_training_parameters['max_epochs'],
    steps_per_epoch=gan_training_parameters['steps_per_epoch'],
    x_val=x_val_combined,
    y_stroke_val=y_stroke_val,
    y_val_raw=y_val_raw
)

# After training completes, save final models and parameters
generator.save(os.path.join(save_path, 'final_generator.keras'))
discriminator.save(os.path.join(save_path, 'final_discriminator.keras'))

with open(os.path.join(save_path, 'train_val_dicts.pkl'), 'wb') as f:
    pickle.dump([train_dict, val_dict], f)

with open(os.path.join(save_path, 'data_parameters.pkl'), 'wb') as f:
    pickle.dump([data_parameters], f)

with open(os.path.join(save_path, 'gan_training_parameters.pkl'), 'wb') as f:
    pickle.dump([gan_training_parameters], f)
"""
with open(os.path.join(save_path, 'generator_parameters.pkl'), 'wb') as f:
    pickle.dump([generator_parameters], f)

with open(os.path.join(save_path, 'discriminator_parameters.pkl'), 'wb') as f:
    pickle.dump([discriminator_parameters], f)
"""
with open(os.path.join(save_path, 'history.pkl'), 'wb') as f:
    pickle.dump([history], f)

# At the end of your script
if epoch_counter.numpy() >= gan_training_parameters['max_epochs']:
    print("✅ Training completed successfully. Cleaning up checkpoints.")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
else:
    print("⚠️ Training did not complete all epochs. Keeping checkpoints for resuming later.")
