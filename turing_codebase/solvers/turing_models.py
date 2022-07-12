import numpy as np
import numba

#  $\partial_t u = D_u (\partial_x^2 + \partial_y^2)u - \alpha u + \epsilon v$
#  $\partial_t v = D_v (\partial_x^2 + \partial_y^2)v -u + \mu v - v^3$
@numba.jit(nopython=True)
def FitzHugh_Nagumo(c, t, f_args):
    alpha, epsilon, mu = f_args
    u = c[0, :, :]
    v = c[1, :, :]
    fu = -alpha * u + epsilon * v
    fv = -u + mu * v - v * v * v
    return np.stack((fu, fv))


# $\partial_t u = D_u (\partial_x^2 + \partial_y^2)u + c_1 -c_0 u + c_3u^2v$
# $\partial_t v = D_v (\partial_x^2 + \partial_y^2)v + c_2 -c_3 u^2 v$
@numba.jit(nopython=True)
def Schnakenberg(c, t, f_args):
    c_0, c_1, c_2, c_3 = f_args
    u = c[0, :, :]
    v = c[1, :, :]
    u2v = (u**2) * v
    fu = c_1 - c_0 * u + c_3 * u2v
    fv = c_2 - c_3 * u2v
    return np.stack((fu, fv))


# $\partial_t u = D_a (\partial_x^2 + \partial_y^2)u + \rho_u \frac{u^2 v}{1 + \kappa_u u^2} - \mu_u u + \sigma_u$
# $\partial_t v = D_s (\partial_x^2 + \partial_y^2)v - \rho_v\frac{u^2 v}{1 + \kappa_u u^2} + \sigma_v$
@numba.jit(nopython=True)
def Koch_Meinhardt(c, t, f_args):
    sigma_u, sigma_v, rho_u, rho_v, kappa_u, mu_u = f_args
    u = c[0, :, :]
    v = c[1, :, :]
    u2 = u**2
    u2v = u2 * v
    u2v_u2 = u2v / (1.0 + kappa_u * u2)
    fu = rho_u * u2v_u2 - mu_u * u + sigma_u
    fv = -rho_v * u2v_u2 + sigma_v
    return np.stack((fu, fv))
