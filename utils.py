from neurodiffeq.conditions import BaseCondition
import torch
import numpy as np


class CustomCondition(BaseCondition):
    r"""A custom condition where the parametrization is custom made by the user to comply with
    the conditions of the differential system.

    :param parametrization:
        The custom parametrization that comlpies with the conditions of the differential system. The first input
        is the output of the neural network. The rest of the inputs of the parametrization are
        the inputs to the neural network with the same order as in the solver.
    :type parametrization: callable
    """

    def __init__(self, parametrization):
        super().__init__()
        self.parameterize = parametrization

    def enforce(self, net, *coords):
        r"""Enforces this condition on a network with `N` inputs

        :param net: The network whose output is to be re-parameterized.
        :type net: `torch.nn.Module`
        :param coordinates: Inputs of the neural network.
        :type coordinates: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`
        .. note::
            This method overrides the default method of ``neurodiffeq.conditions.BaseCondition`` .
            In general, you should avoid overriding ``enforce`` when implementing custom boundary conditions.
        """
        def ANN(*coords):
            r"""The neural netowrk as a function

            :param coordinates: Inputs of the neural network.
            :type coordinates: `torch.Tensor`.
            :return: The output or outputs of the network.
            :rtype: list[`torch.Tensor`, `torch.Tensor`,...] or `torch.Tensor`.
            """
            outs = net(torch.cat([*coords], dim=1))
            out_units = net.NN[-1].out_features
            if out_units > 1:
                outs = [outs[:, index].view(-1, 1) for index in range(out_units)]
            return outs

        return self.parameterize(ANN, *coords)


def reparam_CPL(ANN, z, w_0, w_1):
    r"""The reparametrization of the outputs of the netowork for the CPL model,
    which in this case corresponds to the integrating factor:

    :math:`\displaystyle \tilde{x}\left(z, \omega_0, \omega_1 \right)=
    \exp\left\{\left(1+\omega_0\right)\left[x_{\mathcal{N},1}\left(z\right)
    -x_{\mathcal{N},1}\left(z_0\right)\right] + \omega_1\left[x_{\mathcal{N},2}\left(z\right)
    - x_{\mathcal{N},2}\left(z_0\right)\right]\right\}.`

    :param ANN: The neural network.
    :type ANN: function.
    :param z: The independent variable.
    :type z: `torch.Tensor`.
    :param w_0: The first parameter of the bundle.
    :type w_0: `torch.Tensor`.
    :param w_1: The second parameter of the bundle.
    :type w_1: `torch.Tensor`.
    :return: The re-parameterized output, where the condition is automatically satisfied.
    :rtype: `torch.Tensor`.
    """

    x_1, x_2 = ANN(z)

    z_0s = torch.zeros_like(z, requires_grad=True)

    x_1_0, x_2_0 = ANN(z_0s)

    out = np.e**((1 + w_0)*(x_1 - x_1_0) + w_1*(x_2 - x_2_0))
    return out


class quint_reparams:
    def __init__(self, N_0_abs):
        self.N_0_abs = N_0_abs

    def x_reparam(self, ANN, N_prime, lam_prime, Om_m_0):
        r"""The reparametrization of the output of the netowork assinged to the first dependent variable
        of the differential system of the quintessence model,
        which in this case corresponds to a perturbative reparametrization:

        :math:`\displaystyle \tilde{x}\left(N^{\prime}, \lambda^{\prime}, \Omega_{m,0}^{\Lambda}\right)=
        \left(1-e^{-N^{\prime}}\right)\left(1-e^{-\lambda^{\prime}}\right)
        x_{\mathcal{N}}\left(N^{\prime}, \lambda^{\prime}, \Omega_{m,0}^{\Lambda}\right).`

        :param ANN: The neural network.
        :type ANN: function.
        :param N_prime: The independent variable.
        :type N_prime: `torch.Tensor`
        :param lam_prime: The first parameter of the bundle.
        :type lam_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """

        x_N = ANN(N_prime, lam_prime, Om_m_0)

        out = (1 - torch.exp(-N_prime)) * (1 - torch.exp(-lam_prime)) * x_N
        return out

    def y_reparam(self, ANN, N_prime, lam_prime, Om_m_0):
        r"""The reparametrization of the output of the netowork assinged to the second dependent variable
        of the differential system of the quintessence model,
        which in this case corresponds to a perturbative reparametrization:

        :math:`\displaystyle \tilde{y}\left(N^{\prime}, \lambda^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{y}\left(N^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-N^{\prime}}\right)\left(1-e^{-\lambda^{\prime}}\right)
        y_{\mathcal{N}}\left(N^{\prime}, \lambda^{\prime}, \Omega_{m,0}^{\Lambda}\right),`

        where :math:`\hat{y}` is:

        :math:`\displaystyle
        \hat{y}\left(N^{\prime},\Omega_{m,0}^{\Lambda}\right)=
        \sqrt{\dfrac{\left(1-\Omega_{m,0}^{\Lambda}\right)}{\Omega_{m,0}^{\Lambda}e^{-3\left|N_0\right|\left(N^{\prime}-1\right)}+1-\Omega_{m,0}^{\Lambda}}}.`

        :param ANN: The neural network.
        :type ANN: function.
        :param N_prime: The independent variable.
        :type N_prime: `torch.Tensor`
        :param lam_prime: The first parameter of the bundle.
        :type lam_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """

        y_N = ANN(N_prime, lam_prime, Om_m_0)

        N = (N_prime - 1)*self.N_0_abs

        y_hat = ((1 - Om_m_0)/(Om_m_0*(np.e**(-3*N)) + 1 - Om_m_0)) ** (1/2)

        out = y_hat + (1 - torch.exp(-N_prime)) * (1 - torch.exp(-lam_prime)) * y_N
        return out


class HS_reparams:
    def __init__(self, z_0, alpha):
        self.z_0 = z_0
        self.alpha = alpha

    def x_reparam(self, ANN, z_prime, b_prime, Om_m_0):
        r"""The reparametrization of the output of the netowork assinged to the first dependent variable,
        which in this case corresponds to a perturbative reparametrization:

        :math:`\displaystyle \tilde{x}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        x_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right).`

        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha

        x_N = ANN(z_prime, b_prime, Om_m_0)

        out = (1 - torch.exp(-z_prime)) * (1 - torch.exp(alpha*(-b_prime))) * x_N
        return out

    def y_reparam(self, ANN, z_prime, b_prime, Om_m_0):
        r"""The reparametrization of the output of the netowork assinged to the second dependent variable,
        which in this case corresponds to a perturbative reparametrization:

        :math:`\displaystyle \tilde{y}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{y}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        y_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right),`

        where :math:`\hat{y}` is:

        :math:`\displaystyle
        \hat{y}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)=
        \dfrac{\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}
        +2\left(1-\Omega_{m,0 }^{\Lambda}\right)}{2\left[\Omega_{m,0}^{\Lambda}
        \left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}+1-\Omega_{m,0}^{\Lambda}\right]}.`

        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha
        z = self.z_0 * (1 - z_prime)

        y_N = ANN(z_prime, b_prime, Om_m_0)

        y_hat = (Om_m_0*((1 + z)**3) + 2*(1 - Om_m_0))/(2*(Om_m_0*((1 + z)**3) + 1 - Om_m_0))

        out = y_hat + (1 - torch.exp(-z_prime)) * (1 - torch.exp(alpha*(-b_prime))) * y_N
        return out

    def v_reparam(self, ANN, z_prime, b_prime, Om_m_0):
        r"""The reparametrization of the output of the netowork assinged to the third dependent variable,
        which in this case corresponds to a perturbative reparametrization:

        :math:`\displaystyle \tilde{v}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{v}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        v_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right),`

        where :math:`\hat{v}` is:

        :math:`\displaystyle
        \hat{v}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)=
        =\dfrac{\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}
        +4\left(1-\Omega_{m,0 }^{\Lambda}\right)}{2\left[\Omega_{m,0}^{\Lambda}
        \left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}+1-\Omega_{m,0}^{\Lambda}\right]}.`

        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha
        z = self.z_0 * (1 - z_prime)

        v_N = ANN(z_prime, b_prime, Om_m_0)

        v_hat = (Om_m_0*((1 + z)**3) + 4*(1 - Om_m_0))/(2*(Om_m_0*((1 + z)**3) + 1 - Om_m_0))

        out = v_hat + (1 - torch.exp(-z_prime)) * (1 - torch.exp(alpha*(-b_prime))) * v_N
        return out

    def Om_reparam(self, ANN, z_prime, b_prime, Om_m_0):
        r"""The reparametrization of the output of the netowork assinged to the fourth dependent variable,
        which in this case corresponds to a perturbative reparametrization:

        :math:`\displaystyle \tilde{\Omega}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{\Omega}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        \Omega_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right),`

        where :math:`\tilde{\Omega}` is:

        :math:`\displaystyle
        \tilde{\Omega}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \dfrac{\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}}
        {\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}+1-\Omega_{m,0}^{\Lambda}}.`

        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha
        z = self.z_0 * (1 - z_prime)

        Om_N = ANN(z_prime, b_prime, Om_m_0)

        Om_hat = Om_m_0*((1 + z)**3)/((Om_m_0*((1 + z)**3) + 1 - Om_m_0))

        out = Om_hat + (1 - torch.exp(-z_prime)) * (1 - torch.exp(alpha*(-b_prime))) * Om_N
        return out

    def r_prime_reparam(self, ANN, z_prime, b_prime, Om_m_0):
        r"""The reparametrization of the output of the netowork assinged to the fifth dependent variable,
        which in this case corresponds to a perturbative reparametrization:

        :math:`\displaystyle \tilde{r}^{\prime}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{r}^{\prime}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        r^{\prime}_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right),`

        where :math:`\hat{r}^{\prime}` is:

        :math:`\displaystyle
        \hat{r}^{\prime}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)=
        \dfrac{\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}
        +4\left(1-\Omega_{m,0 }^{\Lambda}\right)}{1-\Omega_{m,0}^{\Lambda}}.`

        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha
        z = self.z_0 * (1 - z_prime)

        r_prime_N = ANN(z_prime, b_prime, Om_m_0)

        r_hat = (Om_m_0*((1 + z)**3) + 4*(1 - Om_m_0))/(1 - Om_m_0)

        if isinstance(r_hat, torch.Tensor):
            r_prime_hat = torch.log(r_hat)
        else:
            r_prime_hat = np.log(r_hat)

        out = r_prime_hat + (1 - torch.exp(-z_prime)) * (1 - torch.exp(alpha*(-b_prime))) * r_prime_N
        return out
