# Import libraries:
import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import diff  # the differentiation operation
from utils import CustomCondition, quint_reparams
from neurodiffeq.solvers import BundleSolver1D
import torch

# Set a fixed random seed:
torch.manual_seed(42)

# Set the range of the independent variable:

N_prime_0 = 0.0
N_prime_f = 1.0

# Set the range of the parameters of the bundle:

Om_m_0_min = 0.1
Om_m_0_max = 0.4

lam_prime_min = 0.0
lam_prime_max = 1.0

# Define the differential equation:

z_0 = 10.0
N_0_abs = np.abs(np.log(1/(1 + z_0)))
lam_max = 3.0


def ODEs_quint(x, y, N_prime, *theta):
    r"""Function that defines the differential equations of the system, by defining the residuals of it. In this case:

    :math:`\displaystyle \mathcal{R}_1\left(\tilde{x},\tilde{y},N^{\prime},\lambda^{\prime}\right)=
    \dfrac{1}{\left|N_0\right|}\dfrac{d\tilde{x}}{dN^{\prime}}
    +3\tilde{x}-3\dfrac{\sqrt{6}}{2}\lambda^{\prime} \tilde{y}^{2}
    -\dfrac{3}{2}\tilde{x}\left(1+\tilde{x}^{2}-\tilde{y}^{2}\right),`

    :math:`\displaystyle \mathcal{R}_2\left(\tilde{x},\tilde{y},N^{\prime},\lambda^{\prime}\right)=
    \dfrac{1}{\left|N_0\right|}\dfrac{d\tilde{y}}{dN^{\prime}}
    +3\dfrac{\sqrt{6}}{2}\tilde{x}\tilde{y}\lambda^{\prime}
    -\dfrac{3}{2}\tilde{y}\left(1+\tilde{x}^{2}-\tilde{y}^{2}\right).`

    :param x: The reparametrized output of the network corresponding to the first dependent variable.
    :type x: `torch.Tensor`
    :param y: The reparametrized output of the network corresponding to the second dependent variable.
    :type y: `torch.Tensor`
    :param N_prime: The independent variable.
    :type N_prime: `torch.Tensor`
    :param theta: The parameters of the bundle.
    :type theta: list[`torch.Tensor`,`torch.Tensor`]
    :return: The residuals of the differential equations.
    :rtype: list[`torch.Tensor`, `torch.Tensor`]
    """
    lam_prime = theta[0]

    res_1 = (diff(x, N_prime)/N_0_abs) + 3*x - lam_max*(np.sqrt(6)/2)*lam_prime*(y ** 2) - (3/2)*x*(1 + (x**2) - (y**2))
    res_2 = (diff(y, N_prime)/N_0_abs) + lam_max*(np.sqrt(6)/2)*lam_prime*(y * x) - (3/2)*y*(1 + (x**2) - (y**2))
    return res_1, res_2


# Define the custom reparametrizations that enforce the initial conditions:

quint = quint_reparams(N_0_abs=N_0_abs)

conditions = [CustomCondition(quint.x_reparam),
              CustomCondition(quint.y_reparam)]


# Define a custom loss function:

def weighted_loss_quint(res, f, t):
    r"""A custom loss function. While the default loss is the sum of the squares of the residuals,
    here a weighting function is added:

    :math:`\displaystyle L\left(\tilde{x},\tilde{y},N^{\prime},\lambda^{\prime}\right)
    =\sum^2_{i=1}\mathcal{R}_i\left(\tilde{x},\tilde{y},N^{\prime},\lambda^{\prime}\right)^2e^{-2N^{\prime}\lambda^{\prime}}`

    :param res: The residuals of the differential equation.
    :type res: `torch.Tensor`.
    :param f: The reparametrized outputs of the networks corresponding to the dependent variables.
    :type f: list[`torch.Tensor`, `torch.Tensor`, ...].
    :type t: The inputs of the neural network: i.e, the independent variable and the parameter of the bundle.
    :param t: list[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`].
    :return: The mean value of the loss across the training points.
    :rtype: `torch.Tensor`.
    """

    N_prime, lam_prime = t[0:2]

    w = 2

    mean_res = torch.mean(((torch.exp(-w * N_prime * lam_prime) * (res ** 2))), dim=0)
    return mean_res.sum()


# Define the ANN based solver:
solver = BundleSolver1D(ode_system=ODEs_quint,
                        conditions=conditions,
                        t_min=N_prime_0, t_max=N_prime_f,
                        theta_min=(lam_prime_min, Om_m_0_min),
                        theta_max=(lam_prime_max, Om_m_0_max),
                        criterion=weighted_loss_quint,
                        )

# Set the amount of interations to train the solver:
iterations = 3000

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
plt.savefig('loss.png')

# Save the neural network:
torch.save(solver._get_internal_variables()['best_nets'], 'nets_Quintessence.ph')
