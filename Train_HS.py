# Import libraries
import matplotlib.pyplot as plt
from neurodiffeq.generators import Generator1D
from neurodiffeq import diff  # the differentiation operation
from neurodiffeq.solvers import BundleSolver1D
from utils import CustomCondition, HS_reparams
import torch

# Set a fixed random seed:
torch.manual_seed(42)

# Set the range of the independent variable:

z_prime_0 = 0.0
z_prime_f = 1.0

# Set the range of the parameters of the bundle:

b_prime_min = 1e-13
b_prime_max = 1.0

Om_m_0_min = 0.1
Om_m_0_max = 0.4

# Define the differential equation:

z_0 = 10.0
b_max = 5.0


def ODEs_HS(x, y, v, Om, r_prime, z_prime, *theta):
    r"""Function that defines the differential equations of the system, by defining the residuals of it. In this case:

    :math:`\displaystyle
    \mathcal{R}_1\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{d\tilde{x}}{dz^{\prime}} + \dfrac{z_{0}}{z_{0}\left(1-z^{\prime}\right)+1}
    \left(-\tilde{\Omega}-2\tilde{v}+\tilde{x}+4\tilde{y}+\tilde{x}\tilde{v}+\tilde{x}^{2}\right),`

    :math:`\displaystyle
    \mathcal{R}_2\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{d\tilde{y}}{dz^{\prime}} - \dfrac{z_{0}}{z_{0}\left(1-z^{\prime}\right)+1}
    \left(\tilde{v}\tilde{x}\Gamma\left(\tilde{r}^{\prime}\right)-\tilde{x}\tilde{y}+4\tilde{y}-2\tilde{y}\tilde{v}\right),`

    :math:`\displaystyle
    \mathcal{R}_3\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{d\tilde{v}}{dz^{\prime}} - \dfrac{z_{0}\tilde{v}}{z_{0}\left(1-z^{\prime}\right)+1}
    \left(\tilde{x}\Gamma\left(\tilde{r}^{\prime}\right)+4-2\tilde{v}\right),`

    :math:`\displaystyle
    \mathcal{R}_4\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{d\tilde{\Omega}}{dz^{\prime}}
    + \dfrac{z_{0}\tilde{\Omega}}{z_{0}\left(1-z^{\prime}\right)+1}\left(-1+2\tilde{v}+\tilde{x}\right),`

    :math:`\displaystyle
    \mathcal{R}_5\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{d\tilde{r}^{\prime}}{dz^{\prime}}
    - \dfrac{z_{0}\Gamma\left(\tilde{r}^{\prime}\right)\tilde{x}}{z_{0}\left(1-z^{\prime}\right)+1},`

    where :math:`\Gamma` is:

    :math:`\displaystyle \Gamma\left(\tilde{r}^\prime\right)=
    \dfrac{\left(e^{\tilde{r}^\prime}+5b^{\prime}\right)\left[\left(e^{\tilde{r}^\prime}+5b^{\prime}\right)^{2}-10b^{\prime}\right]}{20b^{\prime}e^{\tilde{r}^\prime}}.`


    :param x: The reparametrized output of the network corresponding to the first dependent variable.
    :type x: `torch.Tensor`
    :param y: The reparametrized output of the network corresponding to the second dependent variable.
    :type y: `torch.Tensor`
    :param v: The reparametrized output of the network corresponding to the third dependent variable.
    :type v: `torch.Tensor`
    :param Om: The reparametrized output of the network corresponding to the fourth dependent variable.
    :type Om: `torch.Tensor`
    :param r_prime: The reparametrized output of the network corresponding to the fifth dependent variable.
    :type r_prime: `torch.Tensor`
    :param z_prime: The independent variable.
    :type z_prime: `torch.Tensor`
    :param theta: The parameters of the bundle.
    :type theta: list[`torch.Tensor`,`torch.Tensor`]
    :return: The residuals of the differential equations.
    :rtype: list[`torch.Tensor`, `torch.Tensor`,...]
    """
    b_prime = theta[0]

    b = b_max * b_prime
    z = z_0 * (1 - z_prime)
    r = torch.exp(r_prime)

    Gamma = (r + b)*(((r + b)**2) - 2*b)/(4*r*b)

    # Equation System:
    res_1 = diff(x, z_prime) + z_0*(-Om - 2*v + x + 4*y + x*v + x**2)/(z + 1)
    res_2 = diff(y, z_prime) - z_0*(v*x*Gamma - x*y + 4*y - 2*y*v)/(z + 1)
    res_3 = diff(v, z_prime) - z_0*v*(x*Gamma + 4 - 2*v)/(z + 1)
    res_4 = diff(Om, z_prime) + z_0*Om*(-1 + 2*v + x)/(z + 1)
    res_5 = diff(r_prime, z_prime) - z_0*(Gamma*x)/(z + 1)

    return [res_1, res_2, res_3, res_4, res_5]


# Define the custom reparametrizations that enforce the initial conditions:

HS = HS_reparams(z_0=z_0, alpha=1/6)

conditions = [CustomCondition(HS.x_reparam),
              CustomCondition(HS.y_reparam),
              CustomCondition(HS.v_reparam),
              CustomCondition(HS.Om_reparam),
              CustomCondition(HS.r_prime_reparam)]

# Define the generetors of the training points for the indepndent variable:

tgz = Generator1D(32, t_min=z_prime_0, t_max=z_prime_f)

vgz = Generator1D(32, t_min=z_prime_0, t_max=z_prime_f)

# Define the generetors of the training points for the parameters of the bundle:

tgb = Generator1D(32, t_min=b_prime_min, t_max=b_prime_max)

tgO = Generator1D(32, t_min=Om_m_0_min, t_max=Om_m_0_max)

vgb = Generator1D(32, t_min=b_prime_min, t_max=b_prime_max)

vgO = Generator1D(32, t_min=Om_m_0_min, t_max=Om_m_0_max)

# Combine all the generators to create the generator for all the input training points:

train_gen = tgz ^ tgb ^ tgO

valid_gen = vgz ^ vgb ^ vgO


# Define a custom loss function:

def custom_loss_HS(res, f, t):
    r"""A custom loss function.
    In this case a sum of two different loss functions :math:`L_{\mathcal{R}}` and :math:`L_{\mathcal{C}}`.
    The former is the part of the loss that concerns the resiudals.
    While the default would be a sum of the squares of the residuals, here a weighting function is added:

    :math:`\displaystyle
    L_{\mathcal{R}}\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)
    =\sum^5_{i=1}\mathcal{R}_i\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)^2e^{-2z^{\prime}b^{\prime}}.`

    On the other hand, :math:`L_{\mathcal{C}}` is constructed from the relative difference between some
    equations' right and left hand side that must hold true due to symetries that relate the variables of the system.
    In particular:

    :math:`\displaystyle \begin{split}
    L_\mathcal{C}\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},
    \tilde{r}^{\prime}, z^{\prime}, b^{\prime}, \Omega^{\Lambda}_{m,0}\right)=
    &\left(\tilde{\Omega} +\tilde{v}-\tilde{x}-\tilde{y}-1\right)^2\\
    &+ \left\{\dfrac{2\tilde{y}\Omega^\Lambda_{m,0}\left[1+z_{0}\left(1 - z^{\prime}\right)\right]^3}
    {\tilde{\Omega}e^{\tilde{r}^\prime}\left(1-\Omega^\Lambda_{m,0}\right)}
    \left[\dfrac{e^{\tilde{r}^\prime}+5b^{\prime}}{e^{\tilde{r}^\prime}+5b^{\prime}-2}\right]-1\right \}^2\\
    &+ \left\{\dfrac{2\tilde{v}\Omega^\Lambda_{m,0}\left[1+z_{0}\left(1 - z^{\prime}\right)\right]^{3}}
    {\tilde{\Omega}e^{\tilde{r}^\prime}\left(1-\Omega^\Lambda_{m,0}\right )}
    \left[\dfrac{\left(e^{\tilde{r}^\prime}+5b^{\prime}\right)^{2}}
    {\left(e^{\tilde{r}^\prime}+5b^{\prime}\right)^{2}-10b^{\prime}}\right]-1\right\}^2. \end{split}`

    Thus, the final total loss is:

    :math:`\displaystyle L=L_{\mathcal{R}}+L_{\mathcal{C}}.`

    :param res: The residuals of the differential equation.
    :type res: `torch.Tensor`.
    :param f: The reparametrized outputs of the networks corresponding to the dependent variables.
    :type f: list[`torch.Tensor`, `torch.Tensor`, ...].
    :type t: The inputs of the neural network: i.e, the independent variable and the parameter of the bundle.
    :param t: list[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`].
    :return: The mean value of the loss across the training points.
    :rtype: `torch.Tensor`.
    """
    z_prime = t[0]
    b_prime = t[1]
    Om_m_0 = t[2]

    x = f[0]
    y = f[1]
    v = f[2]
    Om = f[3]
    r_prime = f[4]

    b = b_max*b_prime
    z = z_0 * (1 - z_prime)
    r = torch.exp(r_prime)

    w = 2

    loss_R = torch.exp(-w * z_prime * (b_prime - b_prime_min)) * (res ** 2)

    loss_C_1 = (1 - Om + v - x - y)**2

    loss_C_2 = (1 - 2*y*Om_m_0*((1+z)**3)*(r+b)/(r*Om*(1-Om_m_0)*(r+b-2)))**2

    loss_C_3 = (1 - 2*v*Om_m_0*((1+z)**3)*((r+b)**2)/(r*Om*(1-Om_m_0)*((r+b-2)**2)))**2

    loss_C = loss_C_1 + loss_C_2 + loss_C_3

    loss = torch.mean(loss_R + loss_C, dim=0).sum()

    return loss


# Define the ANN based solver:
solver = BundleSolver1D(ode_system=ODEs_HS,
                        conditions=conditions,
                        t_min=z_prime_0, t_max=z_prime_f,
                        train_generator=train_gen,
                        valid_generator=valid_gen,
                        criterion=custom_loss_HS
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
torch.save(solver._get_internal_variables()['best_nets'], 'nets_HS.ph')
