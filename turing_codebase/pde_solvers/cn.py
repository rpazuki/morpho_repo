import numpy as np
import scipy as sp


def integrate_1st_order(N, Δt, Us_ret, A_facts, Bcsrs, Us, kinetics, BC2s, record_steps=1):
    """Solve the AU(n+1) = BU(n) + Δtf(n)

    Args:
        N (int):
        Δt (float):
        Us_ret : array for return
        A_facts: Factorised A
        Bcsrs: csr sparse B matrix
        Us (list of JxJ matrices): Initial state
        kinetics (a vectorised function): Functions that calculate the kinetics terms.
                                        The function signatures must be f(U1, U2, .. Un) for
                                        a model with n node.
        BC2s: Boundary conditions corrections
        record_steps (int, optional): The length of time steps to record a result. Defaults to 1.

    Returns:
        : Us_ret
    """
    record_index = 0
    for t_index in range(1, N + 1):
        kinetics_values = kinetics(*Us)
        Us = [
            A_fac(B.dot(U) + Δt * kinetic + (0.0 if BC2 is None else BC2))
            for A_fac, B, kinetic, U, BC2 in zip(A_facts, Bcsrs, kinetics_values, Us, BC2s)
        ]
        if t_index % record_steps == 0:
            Us_ret[record_index, ...] = np.array(Us)
            record_index += 1

    return Us_ret


def Crank_Nicolson_Euler_Forward_1D_A(rx, Ix):
    """LHS matrix.

    Args:
        rx (float): D Δt / 2 Δx^2
                      Δt = T/N
                      Δx = L/J
        Ix (int): x direction Ix + 1 discrete space points number

    Returns:
        (Ix+1)*(Ix+1) matrix: a matrix that must be added to A (LHS) and subtracted from B (RHS)
    """
    return np.diagflat([-rx] * Ix, -1) + np.diagflat([1.0 + 2.0 * rx] * (Ix + 1)) + np.diagflat([-rx] * Ix, 1)


def Crank_Nicolson_Euler_Forward_1D_B(rx, Ix):
    """RHS matrix.

    Args:
        rx (float): D Δt / 2 Δx^2
                      Δt = T/N
                      Δx = L/J
        Ix (int): x direction Ix + 1 discrete space points number

    Returns:
        (Ix+1)*(Ix+1) matrix: a matrix that must be added to A (LHS) and subtracted from B (RHS)
    """
    return np.diagflat([rx] * Ix, -1) + np.diagflat([1.0 - 2.0 * rx] * (Ix + 1)) + np.diagflat([rx] * Ix, 1)


def Neumann_Boundary_1D(rx, Ix, c=0.0, delta_x=0):
    """Neumann boundary condition: du/dx = 0

    Args:
        rx (float): D Δt / 2 Δx^2
                      Δt = T/N
                      Δx = L/J
        Ix (int): x direction Ix + 1 discrete space points number
        c  (float): constant flux
        delta_x (float)

    Returns:
        ((Ix+1)*(Ix+1) matrix, (Ix+1) vector:
         a matrix that must be added to A (LHS) and subtracted from B (RHS) and
         a correction on the RHS
    """
    if c == 0:
        return (np.diagflat([-rx] + [0] * (Ix - 1) + [-rx]), None)
    else:
        return (
            np.diagflat([-rx] + [0] * (Ix - 1) + [-rx]),
            np.array([-2.0 * rx * c] + [0] * (Ix - 1) + [2.0 * rx * c]) * delta_x,
        )


def Diritchlet_Boundary_1D(rx, Ix, c=0.0, delta_x=0):
    """Diritchlet boundary condition: u = 0

    Args:
        rx (float): D Δt / 2 Δx^2
                      Δt = T/N
                      Δx = L/J
        Ix (int): x direction discrete space points number
        c  (float): constant value at
        delta_x (float)

    Returns:
        ((Ix+1)*(Ix+1) matrix, (Ix+1) vector:
         a matrix that must be added to A (LHS) and subtracted from B (RHS) and
         a correction on the RHS
    """
    if c == 0.0:
        return (0.0, None)
    else:
        return (0.0, np.array([2.0 * rx * c] + [0] * (Ix - 1) + [2.0 * rx * c]))  # np.diagflat( [0 for i in range(J)] )


def Periodic_Boundary_1D(rx, Ix, c=0.0, delta_x=0):
    """Periodic boundary condition: u(0) = u(L)

    Args:
        rx (float): D Δt / 2 Δx^2
                      Δt = T/N
                      Δx = L/J
        Ix (int): x direction discrete space points number
        c  (float): None
        delta_x (float)

    Returns:
        ((Ix+1)*(Ix+1) matrix, (Ix+1) vector:
         a matrix that must be added to A (LHS) and subtracted from B (RHS) and
         a correction on the RHS
    """
    return (np.diagflat([-rx], Ix - 1) + np.diagflat([-rx], 1 - Ix), None)


class Reaction_Diffusion_1D:
    def __init__(
        self,
        Ds,
        N,
        T,
        L,
        Ix,
    ):
        self.Ds = Ds
        self.N = N
        self.T = T
        self.L = L
        self.Ix = Ix
        self.Δt = T / N
        self.Δx = L / Ix
        self.rxs = [D * self.Δt / (2.0 * self.Δx**2) for D in self.Ds]
        self.nodes = len(Ds)


class RD_1D_1st_Order(Reaction_Diffusion_1D):
    def __init__(self, Ds, N, T, L, Ix, boundary_condition=Neumann_Boundary_1D, c=0):
        super().__init__(Ds, N, T, L, Ix)
        BCs = [boundary_condition(rx, Ix, c, self.Δx) for rx in self.rxs]
        self.As = [Crank_Nicolson_Euler_Forward_1D_A(rx, Ix) + BC1 for rx, (BC1, _) in zip(self.rxs, BCs)]
        self.A_facts = [sp.sparse.linalg.factorized(A) for A in self.As]
        self.Bs = [Crank_Nicolson_Euler_Forward_1D_B(rx, Ix) - BC1 for rx, (BC1, _) in zip(self.rxs, BCs)]
        self.Bcsrs = [sp.sparse.csr_matrix(B) for B in self.Bs]

        self.BC2s = [BC2 for _, BC2 in BCs]

    def integrate(self, Us, kinetics, record_steps=1):
        """Solve the AU(n+1) = BU(n) + Δtf(n)

        Args:
            Us (list of JxJ matrices): Initial state
            kinetics (list of functions): Functions that calculate the kinetics terms.
                                          The function signatures must be f(U1, U2, .. Un) for
                                          a model with n node.
            record_steps (int, optional): The length of time steps to record a result. Defaults to 1.

        Returns:
            _type_: _description_
        """
        assert self.nodes == len(Us), f"There must be '{self.nodes}' matrices in Us."
        Us_ret = np.zeros(
            (
                self.N // record_steps,
                self.nodes,
            )
            + Us[0].shape
        )
        return integrate_1st_order(
            self.N, self.Δt, Us_ret, self.A_facts, self.Bcsrs, Us, kinetics, self.BC2s, record_steps
        )


def Crank_Nicolson_Adam_Bashforth_1D_A(rx, Ix, delta_x=0):
    pass


def Crank_Nicolson_Adam_Bashforth_1D_B(rx, Ix, delta_x=0):
    pass


def Crank_Nicolson_Adam_Bashforth_1D_C(rx, Ix, delta_x=0):
    pass


class RD_1D_2nd_Order(Reaction_Diffusion_1D):
    def __init__(
        self,
        Ds,
        N,
        T,
        L,
        Ix,
        boundary_condition=Neumann_Boundary_1D,
    ):
        super().__init__(Ds, N, T, L, Ix)
        self.As = [Crank_Nicolson_Adam_Bashforth_1D_A(rx, Ix) + boundary_condition(rx, Ix) for rx in self.rxs]
        self.Bs = [Crank_Nicolson_Adam_Bashforth_1D_B(rx, Ix) - boundary_condition(rx, Ix) for rx in self.rxs]
        self.Bs = [Crank_Nicolson_Adam_Bashforth_1D_C(rx, Ix) - boundary_condition(rx, Ix) for rx in self.rxs]

    def integrate(self, U1s, U0s, record_steps=1):
        """Solve the AU(n+2) = BU(n+1) + CB(n)+ Δt (3f(n+1) - f(n)),
        using Adams–Bashforth second order time intergration.

        Args:
            U0s (list of JxJ matrices): Initial state
            U1s (list of JxJ matrices): Initial state
            record_steps (int, optional): The length of time steps to record a result. Defaults to 1.

        Returns:
            _type_: _description_
        """
        assert len(self.As) == len(U0s), "There are not equal number of As and U0s."
        assert len(self.As) == len(U1s), "There are not equal number of As and U1s."
        Us_ret = np.zeros(
            (
                self.N // record_steps,
                len(U0s),
            )
            + U0s[0].shape
        )
        record_index = 0
        Fs_n = [self.Fs[i](*U0s) for i, _ in enumerate(self.Fs)]
        for t_index in range(1, self.N + 1):
            Fs_n_1 = [self.Fs[i](*U1s) for i, _ in enumerate(self.Fs)]
            U2s = [
                np.linalg.solve(A, B.dot(U_n_1) + C.dot(U_n) + self.Δt * (3.0 * F_n_1 - F_n))
                for A, B, C, U_n_1, U_n, F_n_1, F_n in zip(self.As, self.Bs, self.Cs, U1s, U0s, Fs_n_1, Fs_n)
            ]
            if t_index % record_steps == 0:
                Us_ret[record_index, ...] = np.array(U2s)
                record_index += 1
            U0s = U1s
            U1s = U2s
            Fs_n = Fs_n_1
        return Us_ret


#################################################################
#   2D
def Crank_Nicolson_Euler_Forward_2D_A(rx, ry, Ix, Jy):
    size = Ix * Jy + Ix + Jy + 1

    return (
        np.diagflat([-rx] * (size - 1), 1)
        + np.diagflat([-ry] * (size - Ix - 1), Ix + 1)
        + np.diagflat([1.0 + 2.0 * rx + 2.0 * ry] * size)
        + np.diagflat([-ry] * (size - Ix - 1), -Ix - 1)
        + np.diagflat([-rx] * (size - 1), -1)
    )


def Crank_Nicolson_Euler_Forward_2D_B(rx, ry, Ix, Jy):
    size = Ix * Jy + Ix + Jy + 1
    return (
        np.diagflat([rx] * (size - 1), 1)
        + np.diagflat([ry] * (size - Ix - 1), Ix + 1)
        + np.diagflat([1.0 - 2.0 * rx - 2.0 * ry] * size)
        + np.diagflat([ry] * (size - Ix - 1), -Ix - 1)
        + np.diagflat([rx] * (size - 1), -1)
    )


def first_order_term_corrections(rx, Ix, Jy):
    return np.diagflat(([0] * Ix + [rx]) * (Jy) + [0] * Ix, 1) + np.diagflat(([0] * Ix + [rx]) * (Jy) + [0] * Ix, 1).T


def Neumann_Boundary_2D(rx, ry, Ix, Jy, cx=0.0, cy=0.0, delta_x=0, delta_y=0):
    """Neumann boundary condition: du/dx, du/dy = 0

        INCOMPELETE

    Args:
        rx (float): D Δt / 2 Δx^2
        ry (float): D Δt / 2 Δx^2
                    Δt = T/N
                    Δx = Lx/Ix
                    Δy = Ly/Jy
        Ix (int): x direction Ix+1 discrete space points number
        Jy (int): y direction Jy+1 discrete space points number

    Returns:
        (IxJy + Ix + Jy)*(IxJy  + Ix + Jy) matrix: A matrix that must be added to A and subtracted from B
    """
    size = Ix * Jy + Ix + Jy + 1
    if cx == 0 and cy == 0:
        return (
            np.diagflat([-rx] * (Jy + 1) + [0] * (size - 2 * Jy - 2) + [-rx] * (Jy + 1))
            + np.diagflat(([-ry] + [0] * (Ix - 1) + [-ry]) * (Jy + 1))
            + first_order_term_corrections(rx, Ix, Jy),
            None,
        )
    else:
        return (
            np.diagflat([-rx] * (Jy + 1) + [0] * (size - 2 * Jy - 2) + [-rx] * (Jy + 1))
            + np.diagflat(([-ry] + [0] * (Ix - 1) + [-ry]) * (Jy + 1))
            + first_order_term_corrections(rx, Ix, Jy),
            np.array(
                [-2.0 * rx * cx * delta_x] * (Jy + 1) + [0] * (size - 2 * Jy - 2) + [2.0 * rx * cx * delta_x] * (Jy + 1)
            )
            + np.array(([-2.0 * ry * cy * delta_y] + [0] * (Ix - 1) + [2.0 * ry * cy * delta_y]) * (Jy + 1)),
        )


def Diritchlet_Boundary_2D(rx, ry, Ix, Jy, cx=0.0, cy=0.0, delta_x=0, delta_y=0):
    """Diritchlet boundary condition: u = 0

    Args:
        rx (float): D Δt / 2 Δx^2
        ry (float): D Δt / 2 Δx^2
                    Δt = T/N
                    Δx = Lx/I
                    Δy = Ly/J
        Ix (int): x direction Ix+1 discrete space points number
        Jy (int): y direction Jy+1 discrete space points number

    Returns:
        (IxJy + Ix + Jy)*(IxJy  + Ix + Jy) matrix: A matrix that must be added to A and subtracted from B
    """
    size = Ix * Jy + Ix + Jy + 1
    if cx == 0.0 and cy == 0.0:
        return (
            np.diagflat([-rx * cx] * (Jy + 1) + [0] * (size - 2 * Jy - 2) + [-rx * cx] * (Jy + 1))
            + np.diagflat(([-ry * cy] + [0] * (Ix - 1) + [-ry * cy]) * (Jy + 1))
            + first_order_term_corrections(rx, Ix, Jy),
            None,
        )
    else:
        return (
            np.diagflat([-rx * cx] * (Jy + 1) + [0] * (size - 2 * Jy - 2) + [-rx * cx] * (Jy + 1))
            + np.diagflat(([-ry * cy] + [0] * (Ix - 1) + [-ry * cy]) * (Jy + 1))
            + first_order_term_corrections(rx, Ix, Jy),
            np.array(
                [2.0 * rx * cx * delta_x] * (Jy + 1) + [0] * (size - 2 * Jy - 2) + [2.0 * rx * cx * delta_x] * (Jy + 1)
            )
            + np.array(([2.0 * ry * cy * delta_y] + [0] * (Ix - 1) + [2.0 * ry * cy * delta_y]) * (Jy + 1)),
        )


def Periodic_Boundary_2D(rx, ry, Ix, Jy, cx=0.0, cy=0.0, delta_x=0, delta_y=0):
    """Periodic boundary condition: u(0) = u(L)

    Args:
        rx (float): D Δt / 2 Δx^2
        ry (float): D Δt / 2 Δx^2
                    Δt = T/N
                    Δx = Lx/I
                    Δy = Ly/J
        Ix (int): x direction Ix+1 discrete space points number
        Jy (int): y direction Jy+1 discrete space points number

    Returns:
        (IxJy + Ix + Jy)*(IxJy  + Ix + Jy) matrix: A matrix that must be added to A and subtracted from B
    """
    return (
        (
            np.diagflat([-ry] * (Ix + 1), Ix * Jy + Jy)
            + np.diagflat(([-rx] + [0] * Ix) * (Jy) + [-rx], Ix)
            + np.diagflat([-ry] * (Ix + 1), Ix * Jy + Jy).T
            + np.diagflat(([-rx] + [0] * Ix) * (Jy) + [-rx], Ix).T
            + first_order_term_corrections(rx, Ix, Jy)
        ),
        None,
    )


class Reaction_Diffusion_2D:
    def __init__(
        self,
        Ds,
        N,
        T,
        Lx,
        Ix,
        Ly,
        Jy,
    ):
        self.Ds = Ds
        self.N = N
        self.T = T
        self.Lx = Lx
        self.Ix = Ix
        self.Ly = Ly
        self.Jy = Jy
        self.Δt = T / N
        self.Δx = Lx / Ix
        self.Δy = Ly / Jy
        self.rxs = [D * self.Δt / (2.0 * self.Δx**2) for D in self.Ds]
        self.rys = [D * self.Δt / (2.0 * self.Δy**2) for D in self.Ds]
        self.nodes = len(Ds)


class RD_2D_1st_Order(Reaction_Diffusion_2D):
    def __init__(self, Ds, N, T, Lx, Ix, Ly, Jy, boundary_condition=Neumann_Boundary_2D, cx=0, cy=0):
        super().__init__(Ds, N, T, Lx, Ix, Ly, Jy)
        BCs = [boundary_condition(rx, ry, Ix, Jy, cx, cy, self.Δx, self.Δy) for rx, ry in zip(self.rxs, self.rys)]
        As = [
            Crank_Nicolson_Euler_Forward_2D_A(rx, ry, Ix, Jy) + BC1 for rx, ry, (BC1, _) in zip(self.rxs, self.rys, BCs)
        ]
        self.A_facts = [sp.sparse.linalg.factorized(A) for A in As]
        Bs = [
            Crank_Nicolson_Euler_Forward_2D_B(rx, ry, Ix, Jy) - BC1 for rx, ry, (BC1, _) in zip(self.rxs, self.rys, BCs)
        ]
        self.Bcsrs = [sp.sparse.csr_matrix(B) for B in Bs]

        self.BC2s = [BC2 for _, BC2 in BCs]

    def integrate(self, Us, kinetics, record_steps=1):
        """Solve the AU(n+1) = BU(n) + Δtf(n)

        Args:
            Us (list of JxJ matrices): Initial state
            kinetics (a vectorised function): Functions that calculate the kinetics terms.
                                          The function signatures must be f(U1, U2, .. Un) for
                                          a model with n node.
            record_steps (int, optional): The length of time steps to record a result. Defaults to 1.

        Returns:
            _type_: _description_
        """
        assert self.nodes == len(Us), f"There must be '{self.nodes}' matrices in Us."
        Us_ret = np.zeros(
            (
                self.N // record_steps,
                self.nodes,
            )
            + Us[0].shape
        )
        return integrate_1st_order(
            self.N, self.Δt, Us_ret, self.A_facts, self.Bcsrs, Us, kinetics, self.BC2s, record_steps
        )
