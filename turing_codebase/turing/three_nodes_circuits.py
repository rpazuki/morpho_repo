from scipy.optimize import fsolve


def act(x, K, n):
    """Activatrion"""
    return 1 / (1 + (K / x) ** n)


def inh(x, K, n):
    """Inhibition"""
    return 1 / (1 + (x / K) ** n)


def create_circuit_3954(
    n, b_A, mu_A, V_A, K_AA, K_AB, K_AC, b_B, mu_B, V_B, K_BA, K_BC, b_C, mu_C, V_C, K_CB, K_CC, single_arg=False
):
    def circuit_3954(A, B, C):
        fA_v = b_A + V_A * act(A, K_AA, n) * inh(B, K_BA, n) - mu_A * A
        fB_v = b_B + V_B * act(A, K_AB, n) * inh(C, K_CB, n) - mu_B * B
        fC_v = b_C + V_C * inh(A, K_AC, n) * inh(B, K_BC, n) * act(C, K_CC, n) - mu_C * C
        return (fA_v, fB_v, fC_v)

    def single(args):
        A, B, C = args
        return circuit_3954(A, B, C)

    if single_arg:
        return single
    else:
        return circuit_3954


def create_circuit_3708(
    n, b_A, mu_A, V_A, K_AB, K_AC, b_B, mu_B, V_B, K_BA, K_BC, b_C, mu_C, V_C, K_CA, K_CB, K_CC, single_arg=False
):
    def circuit_3708(A, B, C):
        fA_v = b_A + V_A * act(B, K_BA, n) * inh(C, K_CA, n) - mu_A * A
        fB_v = b_B + V_B * inh(A, K_AB, n) * act(C, K_CB, n) - mu_B * B
        fC_v = b_C + V_C * inh(A, K_AC, n) * inh(B, K_BC, n) * act(C, K_CC, n) - mu_C * C
        return (fA_v, fB_v, fC_v)

    def single(args):
        A, B, C = args
        return circuit_3708(A, B, C)

    if single_arg:
        return single
    else:
        return circuit_3708


def find_roots(circut_single_arg):
    roots, d, ier, msg = fsolve(circut_single_arg, [10, 10, 10], xtol=1e-10, maxfev=100000, full_output=1)
    # check the solution is valid
    (a_1, b_1, c_1) = circut_single_arg(roots)
    if ier != 1 or a_1 > 1e-8 or b_1 > 1e-8 or c_1 > 1e-8:
        roots, d, ier, msg = fsolve(circut_single_arg, [-1, -1, -1], xtol=1e-10, maxfev=100000, full_output=1)
        # check the solution is valid
        (a_1, b_1, c_1) = circut_single_arg(roots)
        if ier != 1 or a_1 > 1e-8 or b_1 > 1e-8 or c_1 > 1e-8:
            return (roots, -1, msg)
    return (roots, 0, msg)
