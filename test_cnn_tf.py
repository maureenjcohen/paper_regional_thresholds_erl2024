""" Just the CNN architecture itself
    in tensorflow framework"""

# %%
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow import keras
import numpy as np

import custom_metrics

#%%
RSEED = 66
# Seed for random number generator (makes it reproducible)

# %%
# Fake data to test

# %%
def add_cyclic_longitudes(inputs, nlon_wrap=10):
    """
     Extends input data by wrapping (repeating) longitudes at either end
     This is to compensate for the discontinuity in longitude 

     Args:
        inputs: 
            Some dimension?
        nlon_wrap: 
            How many longitudes to wrap at each end
    """
    # inputs: [sample, lat, lon, channel]
    # adding the last 10 lons to the front and first 10 lons to the end
    padded_inputs = tf.concat([inputs, inputs[:, :, :nlon_wrap]], axis=2)
    # padded_inputs = tf.concat([inputs[:, :, -1 * nlon_wrap :], padded_inputs], axis=2)

    return padded_inputs

# %%
class Exponentiate(keras.layers.Layer):
    """Custom layer to exp the sigma and tau estimates inline."""

    def __init__(self, **kwargs):
        super(Exponentiate, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)

# %%
def RegressLossExpSigma(y_true, y_pred):
    """
    Loss function of network

    Negative of the log-likelihood 

    Args:
        y_true: 
            2D tensor of 'real' mu, sigma from CMIP6 selection
        y_pred): 
            2D tensor of predicted mu, sigma from network

    Returns:
        Mean of the negative log-likelihood
    """
    # network predictions
    mu = y_pred[:, 0] # Mean
    sigma = y_pred[:, 1] # Standard deviation

    # normal distribution defined by N(mu,sigma)
    norm_dist = tfp.distributions.Normal(mu, sigma)

    # compute the log as the -log(p)
    # Probability that the true value came from the predicted distribution
    loss = -norm_dist.log_prob(y_true[:, 0])

    return tf.reduce_mean(loss, axis=-1)

# %%
def compile_model(input_data, y_train):

    """
    Convolutional neural network 

    Args:
        input_data: 
            Annual mean anomalies of near-surface air temperature
            from CMIP6 selection, flattened to 1-D tensor
    """
    x_train, target_temps = input_data

    # create input for target temp
    # 'Input' is a tensorflow layer, shape is a single number
    input_target_temp = Input(shape=(1,), name="input_target_temp")

    # create input for maps
    # Shape is a 1-D array
    input_maps = Input(shape=x_train.shape[1:], name="input_maps")

    # Initialise layers with annual mean anomalies
    layers = input_maps

    # First layer is our annual mean anomalies with padded longitudes
    layers = tf.keras.layers.Lambda(add_cyclic_longitudes, name="padded_inputs")(
            layers
        )
    for layer, kernel in enumerate([32,32,32]):
    # Then we add one convolution+one max pooling layer
    # And repeat a total of three times

        # 2D convolution
        layers = tf.keras.layers.Conv2D(
                    filters=kernel,
                    kernel_size=[3,3],
                    use_bias=True,
                    activation="relu",
                    padding="same",
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        seed=RSEED
                    ),
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        seed=RSEED
                    ),
                    name="conv_" + str(layer),
                )(layers)
        # 2D max pooling
        layers = tf.keras.layers.MaxPooling2D(
                (2, 2), padding="same", name="maxpool_" + str(layer)
            )(layers)
    
    # Then flatten
    layers = tf.keras.layers.Flatten(name="flatten_0")(layers)

    # SET OF LAYERS FOR MU (MEAN)
    # Concatenate this with the target temperatures
    layers_mu = tf.keras.layers.Concatenate(axis=-1, name="mu_concat_0")(
        [layers, input_target_temp]
    )

    # Dense layers for the anomalies+target temps
    for layer, nodes in enumerate([10, 10, 10]):
    # One densely connected layer followed by a dropout layer
    # However the dropout rate is 0.0 so I guess nothing gets dropped
    # Repeat 3 times
        ridge_initial = 0.0
        layers_mu = tf.keras.layers.Dense(
            nodes,
            activation="relu",
            kernel_initializer=tf.keras.initializers.RandomNormal(
                seed=RSEED
            ),
            bias_initializer=tf.keras.initializers.RandomNormal(
                seed=RSEED + 1
            ),
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=0.0, l2=ridge_initial
            ),
            name="mu_dense_" + str(layer),
        )(layers_mu)

        layers_mu = tf.keras.layers.Dropout(
            0.0, name="mu_dropout_" + str(layer), seed=RSEED
        )(layers_mu)

    # SET OF LAYERS FOR SIGMA (STDEV)
    # Concatenate layers with target temps, as for layers_mu
    layers_sigma = tf.keras.layers.Concatenate(axis=-1, name="sigma_concat_0")(
    [layers, input_target_temp]
    )
    for layer, nodes in enumerate([10, 10, 10]):
    # Exact same as for layers_mu
        ridge_initial = 0.0
        layers_sigma = tf.keras.layers.Dense(
            nodes,
            activation="relu",
            kernel_initializer=tf.keras.initializers.RandomNormal(
                seed=RSEED
            ),
            bias_initializer=tf.keras.initializers.RandomNormal(
                seed=RSEED + 1
            ),
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=0.0, l2=ridge_initial
            ),
            name="sigma_dense_" + str(layer),
        )(layers_sigma)

        layers_sigma = tf.keras.layers.Dropout(
            0.0, name="sigma_dropout_" + str(layer), seed=RSEED
        )(layers_sigma)

    # final layer until output, concatenate target_temp again
    layers_mu = tf.keras.layers.Concatenate(axis=-1, name="mu_concat_1")(
        [layers_mu, input_target_temp]
    )

    # One additional dense layer for mu with different activation function
    layers_mu = Dense(
        10,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.0),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=RSEED),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            seed=RSEED
        ),
        name="mu_finaldense",
    )(layers_mu)

    layers_sigma = tf.keras.layers.Concatenate(axis=-1, name="sigma_concat_1")(
        [layers_sigma, input_target_temp]
    )

    # Same for sigma, a final dense layer with tanh
    layers_sigma = Dense(
        10,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.0),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=RSEED),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            seed=RSEED
        ),
        name="sigma_finaldense",
    )(layers_sigma)

    # Loss function defined above
    LOSS = RegressLossExpSigma

    y_avg = np.mean(y_train)
    y_std = np.std(y_train)

    mu_z_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(
            seed=RSEED + 100
        ),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            seed=RSEED + 100
        ),
        name="mu_z_unit",
    )(layers_mu)

    mu_unit = tf.keras.layers.Rescaling(
        scale=y_std,
        offset=y_avg,
        name="mu_unit",
    )(mu_z_unit)

    # sigma_unit. The network predicts the log of the scaled sigma_z, then
    # the resclaing layer scales it up to log of sigma y, and the custom
    # Exponentiate layer converts it to sigma_y.
    log_sigma_z_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="log_sigma_z_unit",
    )(layers_sigma)

    log_sigma_unit = tf.keras.layers.Rescaling(
        scale=1.0,
        offset=np.log(y_std),
        name="log_sigma_unit",
    )(log_sigma_z_unit)

    sigma_unit = Exponentiate(
        name="sigma_unit",
    )(log_sigma_unit)

    output_layer = tf.keras.layers.concatenate(
        [mu_unit, sigma_unit], axis=1, name="output_layer"
    )

    # Constructing the model
    model = Model((input_maps, input_target_temp), output_layer)
    try:
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=0.00005
            ),
            loss=LOSS
        )
    except:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
            loss=LOSS
        )

    print(model.summary())

    return model