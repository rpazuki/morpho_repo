

def create_Brusselator(
    A, B, single_arg=False
):
    def Brusselator(u, v):
        u2v = u**2 * v
        fu = A - (B + 1) * u + u2v
        fv = B * u - u2v
        return (fu, fv)

    def single(args):
        u, v = args
        return Brusselator(u, v)

    if single_arg:
        return single
    else:
        return Brusselator
    
    
def create_FitzHugh_Nagumo(
    c_0, c_1, c_2, c_3, single_arg=False
):
    def FitzHugh_Nagumo(u, v):
        fu = mu * u - u * u * u - v + sigma
        fv = b * u - gamma * v
        return (fu, fv)

    def single(args):
        u, v = args
        return FitzHugh_Nagumo(u, v)

    if single_arg:
        return single
    else:
        return FitzHugh_Nagumo
    
def create_Schnakenberg(
    c_0, c_1, c_2, c_3, single_arg=False
):
    def Schnakenberg(u, v):
        u2v = (u**2) * v
        fu = c_1 - c_0 * u + c_3 * u2v
        fv = c_2 - c_3 * u2v
        return (fu, fv)

    def single(args):
        u, v = args
        return Schnakenberg(u, v)

    if single_arg:
        return single
    else:
        return Schnakenberg    
    
def create_Koch_Meinhardt(
    sigma_u, sigma_v, rho_u, rho_v, kappa_u, mu_u, single_arg=False
):
    def Koch_Meinhardt(u, v):
        u2 = u**2
        u2v = u2 * v
        u2v_u2 = u2v / (1.0 + kappa_u * u2)
        fu = rho_u * u2v_u2 - mu_u * u + sigma_u
        fv = -rho_v * u2v_u2 + sigma_v
        return (fu, fv)

    def single(args):
        u, v = args
        return Koch_Meinhardt(u, v)

    if single_arg:
        return single
    else:
        return Koch_Meinhardt        