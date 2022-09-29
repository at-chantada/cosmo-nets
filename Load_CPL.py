# Import libraries:
import matplotlib.pyplot as plt
import numpy as np
from neurodiffeq.solvers import BundleSolution1D
from utils import CustomCondition, reparam_CPL
import torch

# Load the networks:
nets = torch.load('nets_CPL.ph',
                  map_location=torch.device('cpu')  # Needed if trained on GPU but this sciprt is executed on CPU
                  )

# Define the reparametrizations that enforce the initial conditions:
conditions = [CustomCondition(reparam_CPL), ]

# Incorporate the nets and the reparametrizations into a solver:
x = BundleSolution1D(nets, conditions)


# The Hubble parameter as a function of the dependent variables of the system:

def H_CPL(z, w_0, w_1, Om_m_0, H_0, x):
    r"""The Hubble parameter, :math:`H`, as a function of the redshift :math:`z`, the parameters of the funcion,
    and the reparametrized outputs of the network:

    :math:`\displaystyle H=H_0\sqrt{\Omega_{m,0}\left(1+z\right)^3
    +\left(1-\Omega_{m,0}\right)\tilde{x}}.`

    :param z: The redshift.
    :type z: float or `numpy.array`.
    :param w_0: The first parameter of the function.
    :type w_0: float.
    :param w_1: The second parameter of the function.
    :type w_1: float.
    :param Om_m_0: The thrid parameter of the function.
    :type Om_m_0: float.
    :param H_0: The fourth parameter of the function.
    :type H_0: float.
    :param x:
        The reparametrized outputs of the network that represents the dependent variable
        of the differential system of CPL.
    :type x function.
    :return: The value of the Hubble parameter.
    :rtype: float or `numpy.array`.
    """

    xs = x(z, w_0, w_1, to_numpy=True)

    H = H_0*((Om_m_0*((1+z)**3) + (1-Om_m_0)*xs) ** (1/2))

    return H


# Plot the Hubble parameter for different values of the independent variable an its parameters:

zs = np.linspace(0, 3)
for w_0 in np.linspace(-1.4, -0.6, 3):
    for w_1 in np.linspace(-5, 1.7, 2):
        Hs = H_CPL(zs, w_0, w_1, 0.3, 73, x)
        plt.plot(zs, Hs, label=r'$\omega_0={}\;\omega_1={}$'.format(np.round(w_0, 3), np.round(w_1, 3)))

plt.xlabel(r'$z$')
plt.ylabel(r'$H\left[\dfrac{\mathrm{km}/\mathrm{s}}{\mathrm{MPc}}\right]$')
plt.legend(loc='best')
plt.title('The Hubble parameter in CPL' + r'$\;\left(H_0=73 \;\Omega_{m,0}=0.3\right)$')
plt.savefig('H_CPL.png')
