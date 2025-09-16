# Import libraries:
import matplotlib.pyplot as plt
import numpy as np
from neurodiffeq.solvers import BundleSolution1D
from utils import CustomCondition, HS_reparams
import torch

# Load the networks:
nets = torch.load('nets_HS.ph',
                  map_location=torch.device('cpu')  # Needed if trained on GPU but this sciprt is executed on CPU
                  )

# Define the reparametrizations that enforce the initial conditions:

z_0 = 10.0
b_max = 5.0

HS = HS_reparams(z_0=z_0, alpha=1/6)

conditions = [CustomCondition(HS.x_reparam),
              CustomCondition(HS.y_reparam),
              CustomCondition(HS.v_reparam),
              CustomCondition(HS.Om_reparam),
              CustomCondition(HS.r_prime_reparam)]

# Incorporate the nets and the reparametrizations into a solver:

v = BundleSolution1D([nets[2]], [conditions[2]])
r_prime = BundleSolution1D([nets[4]], [conditions[4]])


# The Hubble parameter as a function of the dependent variables of the system:

def H_HS(z, b, Om_m_0, H_0, v, r_prime):
    r"""The Hubble parameter, :math:`H`, as a function of the redshift :math:`z`, the parameters of the funcion,
    and the reprarametrized outputs of the neural networks:

    :math:`\displaystyle H=H^\Lambda_{0}\sqrt{\dfrac{e^{\tilde{r}^\prime}}
    {2\tilde{v}}\left(1-\Omega^\Lambda_{m,0}\right)}.`

    :param z: The redshift.
    :type z: float or `numpy.array`.
    :param b: The first parameter of the function.
    :type b: float.
    :param Om_m_0: The second parameter of the function.
    :type Om_m_0: float.
    :param H_0: The third parameter of the function.
    :type H_0: float.
    :param v:
        The reparametrized output of the network that represents the third dependent variable
        of the differential system of Hu-Sawicki.
    :type v function.
    :param r_prime:
        The reparametrized output of the network that represents the fifth dependent variable
        of the differential system of Hu-Sawicki.
    :type r_prime function.
    :return: The value of the Hubble parameter.
    :rtype: float or `numpy.array`.
    """

    shape = np.ones_like(z)

    zs_prime = 1 - (z/z_0)

    b_prime = b/b_max

    b_primes = b_prime*shape

    Om_m_0s = Om_m_0*shape

    vs = v(zs_prime, b_primes, Om_m_0s, to_numpy=True)

    r_primes = r_prime(zs_prime, b_primes, Om_m_0s, to_numpy=True)

    rs = np.exp(r_primes)

    H = H_0*np.sqrt(rs*(1 - Om_m_0)/(2*vs))

    return H


# Plot the Hubble parameter for different values of the independent variable an its parameters:

zs = np.linspace(0, 3)
for b in np.linspace(0, 5, 3):
    for Om_m_0 in np.linspace(0.1, 0.4, 2):
        Hs = H_HS(zs, b, Om_m_0, 73, v, r_prime)
        plt.plot(zs, Hs,
                 label=r'$b={}$'.format(np.round(b, 3)) + r'$\;\Omega_{m,0}=$' + str(np.round(Om_m_0, 3)))

plt.xlabel(r'$z$')
plt.ylabel(r'$H\left[\dfrac{\mathrm{km}/\mathrm{s}}{\mathrm{MPc}}\right]$')
plt.title('The Hubble parameter in Hu-Sawicki' + r'$\;\left(H^{\Lambda}_0=73 \;\Omega^{\Lambda}_{m,0}=0.3\right)$')
plt.legend(loc='best')
plt.savefig('H_HS.png')
