"""
    gan.py

    This file contains the functions used to generate the generator and discriminator for StyleGAN.

    Requirements:
        - TensorFlow 2.0

    Author: Keith Dao
    Date created: 13/10/2021
    Date last modified: 15/10/2021
    Python version: 3.9.7
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    add,
    AveragePooling2D,
    Conv2D,
    Cropping2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    Layer,
    LeakyReLU,
    Reshape,
    UpSampling2D,
)

# Custom layers
class AdaIN(Layer):
    """Adaptive Instance Normalisation Layer."""

    def __init__(self, epsilon: float = 1e-3):

        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape: list[tf.TensorShape]) -> None:

        dim = input_shape[0][-1]
        if dim == None:
            raise ValueError(
                f"Excepted axis {-1} of the input tensor be defined, but got an input with shape {input_shape}."
            )

        super(AdaIN, self).build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Apply the normalisation formula: gamma * (x - mean) / stddev + beta."""

        x, beta, gamma = inputs

        input_shape = x.shape
        axes = list(range(1, len(input_shape) - 1))
        mean = tf.math.reduce_mean(x, axes, keepdims=True)
        stddev = tf.math.reduce_std(x, axes, keepdims=True) + self.epsilon
        normalised = (x - mean) / stddev

        return normalised * gamma + beta


# Layer blocks
def gen_block(
    input: tf.Tensor,
    style: tf.Tensor,
    noise: tf.Tensor,
    filters: int,
    upSample: bool = True,
) -> tf.Tensor:
    """
    For each block, we want to: (In order)
        - Upscale
        - Conv 3x3
        - Add noise
        - AdaIN
        - LeakyReLU
        - Conv 3x3
        - Add noise
        - AdaIN
        - LeakyReLU
    """

    beta = Dense(filters)(style)
    beta = Reshape([1, 1, filters])(beta)
    gamma = Dense(filters)(style)
    gamma = Reshape([1, 1, filters])(gamma)
    n = Conv2D(filters, kernel_size=1, padding="same")(noise)

    # Begin the generator block
    if upSample:
        out = UpSampling2D(interpolation="bilinear")(input)
        out = Conv2D(filters, kernel_size=3, padding="same")(out)
    else:
        out = Activation("linear")(input)
    out = add([out, n])
    out = AdaIN()([out, beta, gamma])
    out = LeakyReLU(0.01)(out)

    # Compute new beta, gamma and noise
    beta = Dense(filters)(style)
    beta = Reshape([1, 1, filters])(beta)
    gamma = Dense(filters)(style)
    gamma = Reshape([1, 1, filters])(gamma)
    n = Conv2D(filters, kernel_size=1, padding="same")(noise)

    # Continue the generator block
    out = Conv2D(filters, kernel_size=3, padding="same")(out)
    out = add([out, n])
    out = AdaIN()([out, beta, gamma])
    out = LeakyReLU(0.01)(out)

    return out


def disc_block(input: tf.Tensor, filters: int) -> tf.Tensor:
    """
    For each block, we want to: (In order)
        - Conv2D
        - AveragePool2D
        - Conv2D
    """

    # Begin the discriminator block
    out = Conv2D(filters, kernel_size=3, padding="same")(input)
    out = AveragePooling2D()(out)
    out = LeakyReLU(0.01)(out)
    out = Conv2D(filters, kernel_size=3, padding="same")(out)
    out = LeakyReLU(0.01)(out)

    return out


# Optimisers
def get_optimizer(**hyperparameters) -> tf.keras.optimizers.Optimizer:

    return tf.keras.optimizers.Adam(**hyperparameters)


# Models
def get_generator(
    latent_dim: int,
    output_size: int,
    num_filters: int,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
    loss: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(),
) -> tf.keras.Model:

    # Mapping network
    input_mapping = Input(shape=[latent_dim])
    mapping = input_mapping
    mapping_layers = 8
    for _ in range(mapping_layers):
        mapping = Dense(num_filters)(mapping)
        mapping = LeakyReLU(0.01)(mapping)

    # Crop the noise image for each resolution
    input_noise = Input(shape=[output_size, output_size, 1])
    noise = [Activation("linear")(input_noise)]
    curr_size = output_size
    while curr_size > 4:
        curr_size //= 2
        noise.append(Cropping2D(curr_size // 2)(noise[-1]))

    # Generator network
    # Starting block
    curr_size = 4
    input = Input(shape=[1])
    x = Lambda(lambda x: x * 0 + 1)(input)  # Set the constant value to be 1
    x = Dense(curr_size * curr_size * num_filters)(x)
    x = Reshape([curr_size, curr_size, num_filters])(x)
    x = gen_block(x, mapping, noise[-1], num_filters, upSample=False)

    # Add upscaling blocks till the output size is reached
    block = 1
    curr_filters = num_filters
    while curr_size < output_size:
        curr_filters //= 2
        x = gen_block(x, mapping, noise[-(1 + block)], curr_filters)
        block += 1
        curr_size *= 2

    # To greyscale
    x = Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(x)

    generator = tf.keras.Model(
        inputs=[input_mapping, input_noise, input], outputs=x
    )
    generator.compile(optimizer=optimizer, loss=loss)

    return generator


def get_discriminator(
    image_size: int,
    num_filters: int,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
    loss: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(),
) -> tf.keras.Model:

    # Discriminator network
    input = Input(shape=[image_size, image_size, 1])
    x = input
    curr_size = image_size
    while curr_size > 4:
        x = disc_block(x, num_filters // (curr_size // 8))
        curr_size //= 2
    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    x = Activation("sigmoid")(x)

    discriminator = tf.keras.Model(inputs=[input], outputs=x)
    discriminator.compile(optimizer=optimizer, loss=loss)

    return discriminator


# Samples
def generate_samples(
    generator: tf.keras.Model,
    latent_dimension: int,
    sample_size: int,
    img_size: int,
):

    # Generate noise for the generator
    latent_noise = tf.random.normal([sample_size, latent_dimension])
    noise_images = tf.random.normal([sample_size, img_size, img_size, 1])

    samples = generator([latent_noise, noise_images, tf.ones([sample_size, 1])])
    return samples
