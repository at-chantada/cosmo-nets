# Import libraries:
import matplotlib.pyplot as plt
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP
from neurodiffeq import diff  # the differentiation operation
import torch

# Set a fixed random seed:
torch.manual_seed(42)

# Set the range of the independent variable:

z_0 = 0.0
z_f = 3.0

# Set the range of the parameter of the bundle:

Om_m_0_min = 0.1
Om_m_0_max = 0.4


# Define the differential equation:

def ODE_LCDM(x, z):
    r"""Function that defines the differential equation of the system, by defining the residual of it. In this case:

    :math:`\displaystyle \mathcal{R}\left(\tilde{x},z\right)=\dfrac{d\tilde{x}}{dz} - \dfrac{3\tilde{x}}{1+z}.`

    :param x: The reparametrized output of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`.
    :param z: The independent variable.
    :type z: `torch.Tensor`.
    :return: The residual of the differential equation.
    :rtype: list[`torch.Tensor`].
    """
    res = diff(x, z) - 3*x/(1 + z)
    return [res]


# Define the initial condition:
condition = [BundleIVP(t_0=z_0, bundle_conditions={'u_0': 0}), ]


# Define a custom loss function:

def weighted_loss_LCDM(res, x, t):
    r"""A custom loss function. While the default loss is the square of the residual,
    here a weighting function is added:

    :math:`\displaystyle L\left(\tilde{x},z\right)=\mathcal{R}\left(\tilde{x},z\right)^2e^{-2\left(z-z_0\right)}.`

    :param res: The residuals of the differential equation.
    :type res: `torch.Tensor`.
    :param x: The reparametrized output of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`.
    :type t: The inputs of the neural network: i.e, the independent variable and the parameter of the bundle.
    :param t: list[`torch.Tensor`, `torch.Tensor`].
    :return: The mean value of the loss across the training points.
    :rtype: `torch.Tensor`.
    """
    z = t[0]
    w = 2

    loss = (res ** 2) * torch.exp(-w * (z - z_0))
    return loss.mean()


# Define the ANN based solver:
solver = BundleSolver1D(ode_system=ODE_LCDM,
                        conditions=condition,
                        t_min=z_0, t_max=z_f,
                        theta_min=Om_m_0_min,
                        theta_max=Om_m_0_max,
                        loss_fn=weighted_loss_LCDM,
                        )

# Set the amount of interations to train the solver:
iterations = 100000

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
plt.savefig('loss_LCDM.png')

# Save the neural network:
torch.save(solver._get_internal_variables()['best_nets'], 'nets_LCDM.ph')
