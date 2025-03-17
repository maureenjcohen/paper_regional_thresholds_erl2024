""" Just the CNN architecture itself
    in pytorch framework"""

# %%
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# %%
RSEED = 66
# Seed for random number generator (makes it reproducible)

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
    padded_inputs = np.concat([inputs, inputs[:, :, :nlon_wrap]], axis=2)
    # padded_inputs = tf.concat([inputs[:, :, -1 * nlon_wrap :], padded_inputs], axis=2)

    return padded_inputs

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
class ThreshNet(nn.Module):
    """
    A deep learning convolutional neural network
    Associates a global map of an environmental variable (such as air temperature)
    with a regional value for time-until-a-threshold is met (such as years until 1.5 C warming)

    Structure:

    """
    def __init__(self, input_shape, target_temp_shape):
        """
        Initialise ThreshNet
        
        Args:
            None
            
        Returns:
            Nothing 
        """
        super(ThreshNet, self).__init__()
        self.input_target_temp = nn.Linear(target_temp_shape[0], 1)  # target temperature input
        self.input_maps = nn.Conv2d(input_shape[0], 32, kernel_size=(3, 3), padding='same')  # 32 channels for conv1

        # Convolutional Layers
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same')  # Conv2D layer
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same')
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same')

        # MaxPool Layers
        self.pool = nn.MaxPool2d(2, 2)

        # Flatten
        self.flatten = nn.Flatten()

        # Dense layers for mu
        self.dense_mu_1 = nn.Linear(32 * input_shape[1] // 2 * input_shape[2] // 2 + 1, 10)  # Adjust for conv output shape
        self.dense_mu_2 = nn.Linear(10, 10)
        self.dense_mu_3 = nn.Linear(10, 10)

        # Dense layers for sigma
        self.dense_sigma_1 = nn.Linear(32 * input_shape[1] // 2 * input_shape[2] // 2 + 1, 10)
        self.dense_sigma_2 = nn.Linear(10, 10)
        self.dense_sigma_3 = nn.Linear(10, 10)

        # Final layers
        self.final_mu = nn.Linear(10, 10)
        self.final_sigma = nn.Linear(10, 10)

        # For mu and sigma scaling
        self.mu_z_unit = nn.Linear(10, 1)
        self.sigma_z_unit = nn.Linear(10, 1)

    def forward(self, x):
        """
        Forward pass of ThreshNet
        """
        # Padding longitudes (optional, depending on your function)
        x_maps = add_cyclic_longitudes(x_maps)

        # Pass through convolutional layers and maxpooling
        x = self.input_maps(x_maps)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Flatten the output
        x = self.flatten(x)

        # Prepare target_temp
        target_temp = self.input_target_temp(x_target_temp).view(-1, 1)

        # Mu pathway
        mu = torch.cat((x, target_temp), dim=1)
        mu = torch.relu(self.dense_mu_1(mu))
        mu = torch.relu(self.dense_mu_2(mu))
        mu = torch.relu(self.dense_mu_3(mu))

        # Sigma pathway
        sigma = torch.cat((x, target_temp), dim=1)
        sigma = torch.relu(self.dense_sigma_1(sigma))
        sigma = torch.relu(self.dense_sigma_2(sigma))
        sigma = torch.relu(self.dense_sigma_3(sigma))

        # Final layers
        mu = torch.tanh(self.final_mu(mu))
        sigma = torch.tanh(self.final_sigma(sigma))

        # Output layer transformations
        mu_unit = self.mu_z_unit(mu)  # mu transformation
        sigma_unit = self.sigma_z_unit(sigma)  # sigma transformation

        # Rescale mu and sigma
        mu_unit = mu_unit * torch.std(x_target_temp) + torch.mean(x_target_temp)
        sigma_unit = torch.exp(sigma_unit)  # Exponentiate to get sigma

        return mu_unit, sigma_unit
    
def compile_model(input_data, y_train):
    """
    Translate TensorFlow model to PyTorch
    """
    x_train, target_temps = input_data

    # Define model input dimensions based on input data
    input_shape = x_train.shape[1:]  # For 2D images
    target_temp_shape = (1,)  # For target temperature

    model = ThreshNet(input_shape=input_shape, target_temp_shape=target_temp_shape)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    # Loss function (You need to define the loss function)
    # Example: Mean squared error loss
    loss_fn = nn.MSELoss()

    print(model)

    return model, optimizer, loss_fn