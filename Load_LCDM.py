# Import libraries:
import matplotlib.pyplot as plt
import numpy as np
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
import torch

# Load the networks:
nets = torch.load('nets_LCDM.ph',
                  map_location=torch.device('cpu')  # Needed if trained on GPU but this sciprt is executed on CPU
                  )

# Define the reparametrizations that enforce the initial conditions:

z_0 = 0.0

conditions = [BundleIVP(t_0=z_0, bundle_conditions={'u_0': 0}), ]

# Incorporate the nets and the reparametrizations into a solver:
x = BundleSolution1D(nets, conditions)

# The Hubble parameter as a function of the dependent variable of the system:


def H_LCDM(z, Om_m_0, H_0, x):
    r"""The Hubble parameter, :math:`H`, as a function of the redshift :math:`z`, the parameters of the funcion,
    and the reparametrized output of the network:

    :math:`\displaystyle H=H_0\sqrt{\tilde{x}+1-\Omega_{m,0}}.`

    :param z: The redshift.
    :type z: float or `numpy.array`.
    :param Om_m_0: The first parameter of the function.
    :type Om_m_0: float.
    :param H_0: The second parameter of the function.
    :type H_0: float.
    :param x:
        The reparametrized output of the network that represents the dependent variable
        of the differential system of :math:`\Lambda\mathrm{CDM}`.
    :type x function.
    :return: The value of the Hubble parameter.
    :rtype: float or `numpy.array`.
    """

    shape = np.ones_like(z)

    Om_m_0s = Om_m_0*shape

    xs = x(z, Om_m_0s, to_numpy=True)

    H = H_0 * ((xs + (1 - Om_m_0)) ** (1/2))

    return H


# Plot the Hubble parameter for different values of the independent variable an its parameters:

zs = np.linspace(0, 3)
for Om_m_0 in np.linspace(0.1, 0.4, 6):
    Hs = H_LCDM(zs, Om_m_0, 73, x)
    plt.plot(zs, Hs, label=r'$\Omega_{m,0}=$' + str(np.round(Om_m_0, 3)))


plt.xlabel(r'$z$')
plt.ylabel(r'$H\left[\dfrac{\mathrm{km}/\mathrm{s}}{\mathrm{MPc}}\right]$')
plt.legend(loc='best')
plt.title('The Hubble parameter in ' + r'$\Lambda\mathrm{CDM}\;\left(H_0=73\right)$')
plt.savefig('H_LCDM.png')
