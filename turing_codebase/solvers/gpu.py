from numba import cuda, float32
import numpy as np
import math


def integrate(c0, t, dt, n, L, Ds, f, f_args, order):
    threadsperblock = (2, 4, 4)
    blockspergrid_x = math.ceil(c0.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(c0.shape[1] / threadsperblock[1])
    blockspergrid_z = math.ceil(c0.shape[2] / threadsperblock[2])

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    ret = np.zeros((c0.shape[0], c0.shape[1], c0.shape[2], t.shape[0]))

    d_n = cuda.to_device(n)
    d_L = cuda.to_device(L)
    d_Ds = cuda.to_device(Ds)
    d_f_args = cuda.to_device(f_args)
    d_ret = cuda.to_device(ret)
    d_c = cuda.to_device(c0)

    d_dc_arr = define_dc(order, c0)
    t_c = t[0]
    assign_GPU[blockspergrid, threadsperblock](d_ret, d_c, 0)

    RHS_GPU = create_GPU(f)

    for i in range(1, order):
        RHS_GPU[blockspergrid, threadsperblock](d_c, d_n, d_L, d_Ds, d_f_args, d_dc_arr[i])
        forward_GPU(d_c, dt, blockspergrid, threadsperblock, i, *d_dc_arr[1 : i + 1])

    for t_i, t_next in enumerate(t[1:]):
        while t_c < t_next:
            RHS_GPU[blockspergrid, threadsperblock](d_c, d_n, d_L, d_Ds, d_f_args, d_dc_arr[0])
            forward_GPU(d_c, dt, blockspergrid, threadsperblock, order, *d_dc_arr)
            t_c += dt
            # Shift all the elements to left
            d_dc_arr = [d_dc_arr[-1]] + d_dc_arr[:-1]
        assign_GPU[blockspergrid, threadsperblock](d_ret, d_c, t_i + 1)
    d_ret.copy_to_host(ret)
    return ret


def create_GPU(f):
    @cuda.jit
    def RHS_GPU(c0, n, L, Ds, f_args, dc):

        z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
        if z >= c0.shape[0] or x >= c0.shape[1] or y >= c0.shape[2]:
            # Quit if (x, y) is outside of valid dc boundary
            return

        # Periodic boundary condition
        i_prev = (x - 1) % n[0]
        i_next = (x + 1) % n[0]

        j_prev = (y - 1) % n[1]
        j_next = (y + 1) % n[1]

        fuv = f(c0[:, x, y], f_args, z)

        dc[z, x, y] = (Ds[z] * n[z] / L[z]) * (
            c0[z, i_prev, y] + c0[z, i_next, y] + c0[z, x, j_prev] + c0[z, x, j_next] - 4.0 * c0[z, x, y]
        ) + fuv

    return RHS_GPU


@cuda.jit
def assign_GPU(cs, c, index):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return
    cs[z, x, y, index] = c[z, x, y]


@cuda.jit
def forward_Euler_GPU(c, dt, dc):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return
    c[z, x, y] += dc[z, x, y] * dt


@cuda.jit
def forward_Adams_Bashforth_2_GPU(c, dt, dc_1, dc_2):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return

    c[z, x, y] += (1.5 * dc_1[z, x, y] - 0.5 * dc_2[z, x, y]) * dt


@cuda.jit
def forward_Adams_Bashforth_3_GPU(c, dt, dc_1, dc_2, dc_3):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return

    c[z, x, y] += ((23 / 12) * dc_1[z, x, y] - (16 / 12) * dc_2[z, x, y] + (5 / 12) * dc_3[z, x, y]) * dt


@cuda.jit
def forward_Adams_Bashforth_4_GPU(c, dt, dc_1, dc_2, dc_3, dc_4):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return

    c[z, x, y] += (
        (55 / 24) * dc_1[z, x, y] - (59 / 24) * dc_2[z, x, y] + (37 / 24) * dc_3[z, x, y] - (9 / 24) * dc_4[z, x, y]
    ) * dt


@cuda.jit
def forward_Adams_Bashforth_5_GPU(c, dt, dc_1, dc_2, dc_3, dc_4, dc_5):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return

    c[z, x, y] += (
        (1901 / 720) * dc_1[z, x, y]
        - (2774 / 720) * dc_2[z, x, y]
        + (2616 / 720) * dc_3[z, x, y]
        - (1274 / 720) * dc_4[z, x, y]
        + (251 / 720) * dc_5[z, x, y]
    ) * dt


@cuda.jit
def backward_Adams_Moulton_2_GPU(c, dt, dc_1, dc_2):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return

    c[z, x, y] += 0.5 * (dc_1[z, x, y] + dc_2[z, x, y]) * dt


@cuda.jit
def backward_Adams_Moulton_3_GPU(c, dt, dc_1, dc_2, dc_3):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return
    c[z, x, y] += ((5 / 12) * dc_1[z, x, y] + (8 / 12) * dc_2[z, x, y] - (1 / 12) * dc_3[z, x, y]) * dt


@cuda.jit
def backward_Adams_Moulton_4_GPU(c, dt, dc_1, dc_2, dc_3, dc_4):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return

    c[z, x, y] += (
        (9 / 24) * dc_1[z, x, y] + (19 / 24) * dc_2[z, x, y] - (5 / 24) * dc_3[z, x, y] + (1 / 24) * dc_4[z, x, y]
    ) * dt


@cuda.jit
def backward_Adams_Moulton_5_GPU(c, dt, dc_1, dc_2, dc_3, dc_4, dc_5):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    y = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if z >= c.shape[0] or x >= c.shape[1] or y >= c.shape[2]:
        # Quit if (x, y) is outside of valid dc boundary
        return
    # v[z, x, y] += dc[z, x, y]*dt
    c[z, x, y] += (
        (251 / 720) * dc_1[z, x, y]
        + (646 / 720) * dc_2[z, x, y]
        - (264 / 720) * dc_3[z, x, y]
        + (106 / 720) * dc_4[z, x, y]
        - (19 / 720) * dc_5[z, x, y]
    ) * dt


def forward_GPU(c, dt, blockspergrid, threadsperblock, order, *d_dc):
    if order == 1:
        forward_Euler_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    elif order == 2:
        backward_Adams_Moulton_2_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    elif order == 3:
        backward_Adams_Moulton_3_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    elif order == 4:
        backward_Adams_Moulton_4_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    elif order == 5:
        backward_Adams_Moulton_5_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    else:
        assert False


def backward_GPU(c, dt, blockspergrid, threadsperblock, order, *d_dc):
    if order == 1:
        forward_Euler_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    elif order == 2:
        forward_Adams_Bashforth_2_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    elif order == 3:
        forward_Adams_Bashforth_3_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    elif order == 4:
        forward_Adams_Bashforth_4_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    elif order == 5:
        forward_Adams_Bashforth_5_GPU[blockspergrid, threadsperblock](c, dt, *d_dc)
    else:
        assert False


def define_dc(order, c0):
    dc_1 = np.zeros_like(c0)
    d_dc_1 = cuda.to_device(dc_1)
    d_ret = [d_dc_1]

    def add_new(d_ret):
        dc_n = np.zeros_like(c0)
        d_dc_n = cuda.to_device(dc_n)
        d_ret += [d_dc_n]

    if order >= 2:
        add_new(d_ret)
    if order >= 3:
        add_new(d_ret)
    if order >= 4:
        add_new(d_ret)
    if order >= 5:
        add_new(d_ret)

    return d_ret
