import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model.predict import calculate_ssim, visualize_autoencoder, visualize_pixelcnn
from model.train import train_pixelcnn, train_vqvae
import tensorflow as tf
from model.dataset import load_data
from model.modules import AE, get_pixel_cnn, num_embeddings, pixelcnn_input_shape, ssim_loss

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)

# Load the Data.
print("Loading data")
data = load_data()
print("Finished Loading Data")

# Initialize the VQVAE, Print a summary.
print("Loading VQVAE")
vqvae = None
if os.path.exists("vqvae.ckpt"):
    print("Existing model found in vqvae.ckpt, using that")
    vqvae = tf.keras.models.load_model("vqvae.ckpt", compile=False)
else:
    print("vqvae.ckpt not found, creating new model")
    vqvae = AE()
    vqvae.compile(optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'],
                  loss=ssim_loss)
    print(vqvae.encoder.summary())
    print(vqvae.vq.summary())
    print(vqvae.decoder.summary())

    # Begin model training and validation.
    print("Beginning VQVAE Fitting")
    training_data = (tf.concat([data["train"], data["validate"]], axis=0))
    model = train_vqvae(vqvae, training_data)
    predictions = model.predict(data["validate"][0:5])
    print("Finished VQVAE Fitting")

    # Save model.
    print("Saving VQVAE")
    model.save("vqvae.ckpt")
    print("VQVAE Saved to vqvae.ckpt")
print("Finished Loading VQVAE")


# Initialize the PixelCNN, Print a summary.
print("Loading PixelCNN")
pixelcnn = None
if os.path.exists("pixelcnn.ckpt"):
    print("Existing model found in pixelcnn.ckpt, using that")
    pixelcnn = tf.keras.models.load_model("pixelcnn.ckpt")
else:
    print("pixelcnn.ckpt not found, creating new model")
    pixelcnn = get_pixel_cnn(kernel_size=max(
        pixelcnn_input_shape[0], pixelcnn_input_shape[1]), input_shape=pixelcnn_input_shape[0:2])

    pixelcnn.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])

    if not os.path.exists("pixelcnn.ckpt"):
        pixelcnn.predict(tf.random.uniform(shape=(1, *pixelcnn_input_shape[0:2]),
                         dtype=tf.int64, maxval=num_embeddings))

    print("Beginning PixelCNN Fitting")
    training_data = tf.concat([data["train"], data["validate"]], axis=0)
    pixelcnn = train_pixelcnn(pixelcnn, training_data, vqvae)
    print("Finished PixelCNN Fitting")

    print("Saving PixelCNN")
    pixelcnn.save("pixelcnn.ckpt")
    print("PixelCNN saved to pixelcnn.ckpt")
print("Loaded PixelCNN")

print("Visualizing VQVAE")
visualize_autoencoder(vqvae, data["test"], 6)
calculate_ssim(vqvae, data["test"])

print("Visualizing PixelCNN")
visualize_pixelcnn(pixelcnn, vqvae, 6)