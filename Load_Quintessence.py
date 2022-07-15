# Import libraries:
import matplotlib.pyplot as plt
import numpy as np
from neurodiffeq.solvers import BundleSolution1D
from utils import CustomCondition, quint_reparams
import torch

z_0 = 10
N_0_abs = np.abs(np.log(1/(1 + z_0)))
lam_max = 3
nets = torch.load('nets_Quintessence.ph',
                  map_location=torch.device('cpu')  # Needed if trained in GPU but this sciprt is executed in CPU
                  )


# Define the reparametrizations that enforces the initial conditions:

quint = quint_reparams(N_0_abs=N_0_abs)

conditions = [CustomCondition(quint.x_reparam),
              CustomCondition(quint.y_reparam)]

x = BundleSolution1D([nets[0]], [conditions[0]])
y = BundleSolution1D([nets[1]], [conditions[1]])


def H_quint(z, lam, Om_m_0, H_0, x, y):
    r"""The Hubble parameter, :math:`H`, as a function of the redshift :math:`z`, the parameters of the funcion,
    and the reprarametrized outputs of the neural networks:

    :math:`\displaystyle H=H^{\Lambda}_0\sqrt{\dfrac{\Omega_{m,0}^{\Lambda}\left(1+z\right)^3}
    {1-\tilde{x}^2-\tilde{y}^2}}.`

    :param z: The redshift.
    :type z: float or `numpy.array`.
    :param lam: The first parameter of the function.
    :type lam: float.
    :param Om_m_0: The second parameter of the function.
    :type Om_m_0: float.
    :param H_0: The third parameter of the function.
    :type H_0: float.
    :param x:
        The reparametrized output of the network that represents the first dependent variable
        of the differential system of Quintessence.
    :type x function.
    :param y:
        The reparametrized output of the network that represents the second dependent variable
        of the differential system of Quintessence.
    :type y function.
    :return: The value of the Hubble parameter.
    :rtype: float or `numpy.array`.
    """

    shape = np.ones_like(z)

    Ns = np.log(1/(1 + z))

    N_primes = (Ns/N_0_abs) + 1

    lam_prime = lam/lam_max

    lam_primes = lam_prime*shape

    Om_m_0s = Om_m_0*shape

    xs = x(N_primes, lam_primes, Om_m_0s, to_numpy=True)

    ys = y(N_primes, lam_primes, Om_m_0s, to_numpy=True)

    H = H_0*((Om_m_0 * ((1 + z) ** 3))/(1 - (xs ** 2) - (ys ** 2))) ** (1/2)

    return H


zs = np.linspace(0, 3)
for lam in np.linspace(0, 3, 3):
    for Om_m_0 in np.linspace(0.1, 0.4, 2):
        Hs = H_quint(zs, lam, Om_m_0, 73, x, y)
        plt.plot(zs, Hs,
                 label=r'$\lambda={}$'.format(np.round(lam, 3)) + r'$\;\Omega_{m,0}=$' + str(np.round(Om_m_0, 3)))

plt.xlabel(r'$z$')
plt.ylabel(r'$H\left[\dfrac{\mathrm{km}/\mathrm{s}}{\mathrm{MPc}}\right]$')
plt.title('The Hubble parameter in Quintessence' + r'$\;\left(H^{\Lambda}_0=73 \;\Omega^{\Lambda}_{m,0}=0.3\right)$')
plt.legend(loc='best')
plt.savefig('H_Quintessence.png')
