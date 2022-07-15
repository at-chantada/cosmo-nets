# Import libraries
from neurodiffeq import diff
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.networks import FCNN    # fully-connect neural network
import torch
import matplotlib.pyplot as plt
from utils import CustomCondition, reparam_CPL
from neurodiffeq.generators import Generator1D

# Set a fixed random seed:
torch.manual_seed(42)

# Set the range of the independent variable:
z_0 = 0.0
z_f = 3.0


# Define the differential equation:
def ODE_CPL(x_DE, z, w_0, w_1):
    r"""Function that defines the differential equation of the system, by defining the residual of it. In this case:

    :math:`\displaystyle \mathcal{R}\left(\tilde{x}_{DE},z,\omega_0,\omega_1\right)=\dfrac{d\tilde{x}_{DE}}{dz}
    - \dfrac{3\tilde{x}_{DE}}{1+z}\left(1+\omega_0 + \dfrac{\omega_1 z}{1+z}\right).`

    :param x_DE: The reparametrized outputs of the network corresponding to the dependent variable.
    :type x_DE: `torch.Tensor`
    :param z: The independent variable.
    :type z: `torch.Tensor`
    :param w_0: The first parameter of the bundle.
    :type w_0: `torch.Tensor`
    :param w_1: The second parameter of the bundle.
    :type w_1: `torch.Tensor`
    :return: The residual of the differential equation.
    :rtype: list[`torch.Tensor`]
    """

    w = w_0 + (w_1*z/(1 + z))

    res = diff(x_DE, z) - 3*((1 + w)/(1 + z))*x_DE
    return [res]


# Define the reparametrization that enforces the initial condition:

condition = [CustomCondition(reparam_CPL)]

# Define the custom netowrk architecture:
nets = [FCNN(n_input_units=1, n_output_units=2, hidden_units=(32, 32),)]

# Define the generetors of the training points for the indepndent variable:
t_z = Generator1D(32, z_0, z_f, method='equally-spaced-noisy')
v_z = Generator1D(32, z_0, z_f, method='equally-spaced')

# Define the generetors of the training points for the parameters of the bundle:
w_0_g = Generator1D(2, -1, -0.8, method='equally-spaced')
w_1_g = Generator1D(2, -0.6, 0.0, method='equally-spaced')

# Combine all the generators to create the generator for all the input training points:
train = t_z ^ w_0_g ^ w_1_g
valid = v_z ^ w_0_g ^ w_1_g

# Define the ANN based solver:
solver = BundleSolver1D(ode_system=ODE_CPL, conditions=condition, t_min=z_0, t_max=z_f,
                        train_generator=train, valid_generator=valid, nets=nets)

# Set the amount of interations to train the solver:
iterations = 10000

# Start training:
solver.fit(iterations)


# Plot the loss during training, and save it:
loss = solver.metrics_history['train_loss']
plt.plot(loss, label='training loss')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.suptitle('Total loss during training')
plt.savefig('loss_CPL.png')

# Save the neural network:
torch.save(solver._get_internal_variables()['best_nets'], 'nets_CPL.ph')
