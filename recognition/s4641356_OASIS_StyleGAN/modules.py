import tensorflow as tf
import numpy as np
import csv

class adaIN(tf.keras.layers.Layer):
    """
    Custom Keras Neural Netork layer to conduct adaptive instance normalization
    As specified by StyleGAN.
    """
    def __init__(self, **kwargs) -> None:
        """
        Instantiate new adaIn layer
        """
        super().__init__(**kwargs)

    def build(self, input_shape: np.array) -> None:
        """
        Called automatically on first 'call' passing the shape of input to
        allow weights to be allocated lazily. Validates the applied tensor shape

        Raises:
            ValueError: If the second input tensors do not posess the correct 
                    number of scaling values (one per channel)

        Args:
            input (np.array): Shape of input passed to layer
        """
        if not (
                (input_shape[0][3] == input_shape[1][1]) and
                (input_shape[0][3] == input_shape[2][1])
                ):
            raise ValueError(
                    "Scale and Bias tensors must have exacty one value per " / 
                    "channel of image input, recieved Image: {}, Scales: {}," /
                    " Biases: {}".format(
                        input_shape[0],input_shape[1], input_shape[2]
                        )
                    )

    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Performs deterministic adaIN

        Args:
            input (list[tf.Tensor,tf.Tensor, tf.Tensor]): list of tensors, 
                    the first is the working image layer, the second is a tensor
                    containing the coresponding scales for each channel, the 
                    third is a tensor containing the corresponding biases for 
                    each channel.

        Returns:
            tf.Tensor: image layer scaled and biased 
                    (same dimensions as first input tensor)
        """

        x,yscale, ybias = input
        #x is 4 dimensional - channel last, add axes to leverage broadcasting
        yscale = yscale[:,tf.newaxis,tf.newaxis,:] 
        ybias = ybias[:,tf.newaxis,tf.newaxis,:]
        mean = tf.math.reduce_mean(tf.math.reduce_mean(x,axis=1),axis=1)[
                :,tf.newaxis,tf.newaxis,:
                ] 
        std = tf.math.reduce_std(tf.math.reduce_std(x,axis=1),axis=1)[
                :,tf.newaxis,tf.newaxis,:
                ]

        return (yscale*(x-mean)/std) + ybias

class addNoise(tf.keras.layers.Layer):
    """
    Custom Keras Neural Network layer to add in specified noise 
            scaled by a learnt factor
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Instantiate new addNoise layer
        """
        super().__init__(**kwargs)

    def build(self, input_shape: np.array) -> None:
        """
        Called automatically on first 'call' passing the shape of input
        To allow weights to be allocated lazily. 
        Validates the applied tensor shape

        Raises:
            ValueError: If the two input tensors do not have matching dimension

        Args:
            input (np.array): Shape of input passed to layer
        """
        if not (input_shape[0] == input_shape[1]):
            raise ValueError(
                    "Inputs must be of same shape, recieved the following: " \
                    "Input 1: {}, Input 2: {}".format(
                        input_shape[0],input_shape[1]
                        )
                    )

        #This layer's weights are the scaling of the noise per channel
        self.noise_weight = self.add_weight( #inherited from keras Layer class
                shape = (1,1,input_shape[0][3]), 
                initializer = tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=1.0
                    ),
                 name = "noise_weight") 

    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Preforms the layer's desired operation using trained weights

        Args:
            input (list[tf.Tensor,tf.Tensor]): List of two Tensors, the 
                    first is the current working image layer, and the second 
                    is the corresponding matrix of noise to add to it. 
                    The two tensors must have matching dimensions

        Returns:
            tf.Tensor: The image tensor with the noise scaled by learnt 
                    weight added to it
        """

        x,noise = input
        return x + (self.noise_weight*noise)

class StyleGAN():
    """
    Class representing a Generational Adversarial Network based off of 
    StyleGAN-1. Combines a generator and a discriminator model with custom 
    training paridigm. Does not directly subclass keras.Model as this does 
    not allow for unsupervised learning. Instead we manually conduct our own
    custom training paradigm as definied in train.py
    """
    METRICS = [ #GAN training metrics
            "discrim_loss_real", 
            "discrim_loss_fake",
            "gen_loss"
            ]  
    GEN_LEARN_RATE =  0.00025
    DISCRIM_LEARN_RATE = 0.0002


    def __init__(self, 
            output_res: int = 256, 
            start_res: int = 4, 
            latent_dim: int = 512, 
            existing_model_folder: str = None
            ) -> None:
        """
            Instantiate a new StyleGAN

        Args:
            output_res (int, optional): Side length of output images in pixels. 
                    Precondition: positive power of 2. Defaults to 256.
            start_res (int, optional): Side length of first iamge generation 
                    layer. Precondition: positive power of 2. Defaults to 4.
            latent_dim (int, optional): Dimension of latent space. Making this 
                    a power of 2 is reccommended. Precondition: positive. 
                    Defaults to 512.
            existing_model_folder (str,optional): Filepath of existing 
                    styleGANModel. If this is not None the other parameters are 
                    ignored and existing styleGAN is loaded instead. 
                    Defaults to None
        """
        super(StyleGAN, self).__init__()
        if existing_model_folder is not None:
            self.load_model(existing_model_folder)
        else:
            self._make_model(output_res, start_res, latent_dim,
                     0, 0, None, None)
            

    def _make_model(self, 
            output_res: int, 
            start_res: int,
            latent_dim: int, 
            trained_epochs: int, 
            trained_mean:int, 
            generator: tf.keras.Model, 
            discriminator: tf.keras.Model
            ) -> None:
        """
        Constructs a StyleGAN, setting the specified parameters, 
        and linking a generator and discriminator.

        Args:
            output_res (int): Side length of images to be output by generator. 
                    Precondition: positive power of 2.
            start_res (int): Side length of the constant tensor to which the 
                    generator upscales and convolves into an image. 
                    Precondition: positive power of 2.
            latent_dim (int): Dimension of the latent space which the generator 
                    draws from. This also specifies the dimension of the adaIN 
                    parameter tensors and thus the number of filters per 
                    generator block.  Making this a power of 2 is reccommended. 
                    Precondition: positive.
            trained_epochs (int): Number of epochs to which the constructed 
                    model has been previously trained.
            trained_mean (int): The mean used in normalising the data this 
                    model has been trained on
            generator (tf.keras.Model): Model specifiying the architecture and 
                    weights to use for the generator. Set this to None to 
                    generate a new generator architecture with newly 
                    initialised weights.
            discriminator (tf.keras.Model): Model specifiying the architecture 
                    and weights to use for the discriminator. Set this to None 
                    to generate a new discriminator architecture with newly 
                    initialised weights.
        """

        #initialise parameters
        self.output_res = output_res
        self.start_res = start_res
        self.latent_dim = latent_dim

        #initialise generator
        self.generator = generator
        if generator is None:
            self.generator = self.get_generator()

        #Starting with 1's produced more stable learning than starting with 0's
        self.generator_base = tf.ones(shape = (
                1, self.start_res//2, self.start_res//2, self.latent_dim
                ) )

        #initialise discriminator
        self.discriminator = discriminator
        if discriminator is None:
            self.discriminator = self.get_discriminator()
 
        #Linked generator discriminator allows loss to be applied to generator
        self.gan = tf.keras.Model(
                self.generator.input, 
                self.discriminator(self.generator.output), 
                name = "StyleGAN")
        self.gan.summary()

        #Training optimizers. Loss is softplus implemented manually in train.py
        self.discriminator_optimiser = tf.keras.optimizers.Adam(
                StyleGAN.DISCRIM_LEARN_RATE
                )
        self.generator_optimiser = tf.keras.optimizers.Adam(
                StyleGAN.GEN_LEARN_RATE
                )

        self.epochs_trained = trained_epochs
        self._data_mean = trained_mean


    def get_generator(self) -> tf.keras.Model:
        """
        Creates a generator model inline with the StyleGAN framework        

        Returns:
            tf.keras.Model: uncompiled generator model
        """
        
        generator_latent_input = tf.keras.layers.Input(
                shape = (self.latent_dim,), 
                name = "Latent_Input"
                )

        #Latent Feature Generation
        z = generator_latent_input
        for i in range(8):
            z = tf.keras.layers.Dense(
                    self.latent_dim, name = "feature_gen_dense_{}".format(i+1)
                    )(z)
            z = tf.keras.layers.LeakyReLU(
                    0.2, name = "feature_gen_leakyrelu_{}".format(i+1)
                    )(z)

        w = z
        adaIN_scales = tf.keras.layers.Dense(
                self.latent_dim, name = "adain_scales"
                )(w)
        adaIN_biases = tf.keras.layers.Dense(
                self.latent_dim, name = "adain_biases"
                )(w)

        #Generation blocks for each feature scale
        curr_res = self.start_res
        
        #using 'tensor =' to give a constant doesn't allow for dynamic
        #batch size (you must specify the length of that dimenstion unless 
        #you hack something using a Lambda layer). This "Input" will 
        #always be fixed in a wrapper calling generator. 
        constant_input = tf.keras.layers.Input(
                shape = (self.start_res//2,self.start_res//2,self.latent_dim), 
                name = "Constant_Initial_Image"
                ) 
       
        x = constant_input
        generator_noise_inputs = [] #keep input handles for model return
        while curr_res <= self.output_res:

            #Each resolution needs an appropriately sized noise inputs
            layer_noise_inputs = (
                    tf.keras.layers.Input(
                        shape = (curr_res,curr_res,self.latent_dim), 
                        name = "{0}x{0}_Noise_Input_1".format(curr_res)
                        ),
                    tf.keras.layers.Input(
                        shape = (curr_res,curr_res,self.latent_dim), 
                        name = "{0}x{0}_Noise_Input_2".format(curr_res)
                        )
                    )
            generator_noise_inputs += list(layer_noise_inputs)
  
            x = tf.keras.layers.UpSampling2D(
                    size=(2, 2), name = "Upsample_to_{0}x{0}".format(curr_res)
                    )(x)
            x = addNoise(name = "{0}x{0}_Noise_1".format(curr_res))(
                    [x,layer_noise_inputs[0]]
                    )
            x = adaIN(name = "{0}x{0}_adaIN_1".format(curr_res))(
                    [x,adaIN_scales,adaIN_biases]
                    )
            x = tf.keras.layers.Conv2D(
                    self.latent_dim, 
                    kernel_size=3, 
                    padding = "same", 
                    name = "{0}x{0}_2D_deconvolution".format(curr_res)
                    )(x)
            x = addNoise(name = "{0}x{0}_Noise_2".format(curr_res))(
                    [x,layer_noise_inputs[1]]
                    )
            x = adaIN(name = "{0}x{0}_adaIN_2".format(curr_res))(
                    [x,adaIN_scales,adaIN_biases]
                    )

            curr_res = curr_res*2
        
        output_image = tf.keras.layers.Conv2D(1, 
                kernel_size=3, 
                padding = "same",
                name = "Final_Image".format(curr_res)
                )(x)

        return tf.keras.Model(
                inputs = (
                    [generator_latent_input] + 
                    generator_noise_inputs + 
                    [constant_input]
                    ),
                outputs = output_image, 
                name = "Generator"
                )
    
    def get_discriminator(self) -> tf.keras.Model:
        """
        Creates a discriminator model inline with the StyleGAN framework        

        Returns:
            tf.keras.Model: uncompiled discriminator model
        """
        discriminator_input = tf.keras.layers.Input(
                shape = (self.output_res,self.output_res,1), 
                name = "Discriminator_Input"
                )#note we expect greyscale images

        #Feature analysis blocks, perform convolution on decreasing image 
        #resolution (and hence filter increasingly macroscopic features)
        current_res = self.output_res
        x = discriminator_input
        while current_res > 4:
            x = tf.keras.layers.Conv2D(
                    self.latent_dim*4//current_res,
                    kernel_size=3,
                    padding = "same", 
                    name = "{0}x{0}_2D_convolution_1".format(current_res)
                    )(x)
            x = tf.keras.layers.Conv2D(
                    self.latent_dim*4//current_res,
                    kernel_size=3,
                    padding = "same", 
                    name = "{0}x{0}_2D_convolution_2".format(current_res)
                    )(x)
            x = tf.keras.layers.LeakyReLU(
                    0.2,name = "{0}x{0}_leaky_reLU_1".format(current_res)
                    )(x)
            x = tf.keras.layers.MaxPooling2D(
                    (2, 2), name = "{0}x{0}_image_reduction".format(current_res)
                    )(x)
            current_res = current_res//2

        #Flatten and compile features for discrimination
        x = tf.keras.layers.Conv2D(
                self.latent_dim, 
                kernel_size=3, 
                padding = "same", 
                name = "{0}x{0}_2D_convolution_1".format(current_res)
                )(x)
        x = tf.keras.layers.Conv2D(
                self.latent_dim, 
                kernel_size=3, 
                padding = "same", 
                name = "{0}x{0}_2D_convolution_2".format(current_res)
                )(x)
        x = tf.keras.layers.LeakyReLU(
                0.2,name = "{0}x{0}_Leaky_ReLU".format(current_res)
                )(x)
        x = tf.keras.layers.Flatten(name = "Flatten")(x)
        
        #Final decision. the higher the value, the more 'real' an image
        x = tf.keras.layers.Dense(1, name = "Discriminate")(x) 

        return tf.keras.Model(inputs = discriminator_input, 
                outputs = x, 
                name = "Discriminator"
                )
            

    def __call__(self, inputs: list[np.array]) -> np.array:
        """
        Allows the StyleGAN to be used as a functional, takes in a latent 
        vector and an appropriate set of noise matrices, and returns the 
        (normalized) image data of a generated image

        Args:
            inputs (list[np.array]): Set of inputs for the generator: 
                    Latent vector followed by the ordered set of pairs 
                    of noise inputs.

        Returns:
            np.array: 2D array holding data for image generated by styleGAN
        """
        return self.generator(inputs + [np.repeat(
                    self.generator_base, inputs[0].shape[0], axis = 0
                    ) ] )

    def save_model(self, folder: str) -> None:
        """
        Saves styleGAN to a targetted folder. Separately saves the keras model 
        containing the discriminator and generator, with their respective 
        weights. This allows training to continue to be split after a reload, 
        with these two components being able to be relinked with weights 
        carrying over upon loading. Also saves the attributes used surrounding 
        these two models, which are useful in designing/conducting training 
        paradigm.

        Args:
            folder (str): path to folder to which the styleGAN should be saved
        """
        #save model parameters
        with open(folder + 'param.csv', mode = 'w', newline='') as f:
            csv.writer(f).writerow([self.output_res,
                    self.start_res,
                    self.latent_dim,
                    self.epochs_trained,
                    self._data_mean
                    ]) 

        #save model architecture and assotated weights
        self.discriminator.save(folder + "discriminator")
        self.generator.save(folder + "generator")

    def load_model(self, folder :str) -> None:
        """
        Loads a styleGAN from a specified folder. Model will posess weights and 
        relevent details of any prior training conducted on it.

        Args:
            folder (str): folder containing the styleGAN to load.
        """

        #Load model Parameters
        params = None
        with open(folder + 'param.csv', mode = 'r') as f:
            params = next(csv.reader(f)) #param csv should be a single row

        #load model architecture and assotiated weights
        discriminator = tf.keras.models.load_model(folder + "discriminator")
        generator = tf.keras.models.load_model(folder + "generator")

        self._make_model(
                int(params[0]),
                int(params[1]),
                int(params[2]),
                int(params[3]),
                float(params[4]),
                generator,
                discriminator
                )
                
        print("Successfully found and loaded StyleGAN located in \"{}\"".format(
                folder))

    def track_mean(self, mean: float) -> None:
        """
        Stash the mean used to center the data the styleGAN is/was trained on.
        Allows generation of images from this model's output without maintaining
        a separate instance of the data loader used to retrieve the training 
        data.

        Args:
            mean (float): mean used to center the data used to train styleGAN 
                    instance
        """
        self._data_mean = mean

    def get_mean(self) -> float:
        """
        Returns mean used to center the data used to train styleGAN instance

        Returns:
            float: mean used to center the data used to train styleGAN instance
        """
        return self._data_mean
        
    def get_generator_base(self) -> tf.Tensor:
        """
        Return constant tensor needed to pass into generator as base to convolve 
        image from

        Returns:
            tf.Tensor: constant generator base tensor
        """
        return self.generator_base
