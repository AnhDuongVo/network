import numpy as np
from numpy.random import default_rng


def generate_connections(N_tar, N_src, p, same=False):
    """
    connect source to target with probability p
      - if populations SAME, avoid self-connection
      - if not SAME, connect any to any avoididing multiple
    return list of sources i and targets j
    """
    nums = np.random.binomial(N_tar - 1, p, N_src)
    i = np.repeat(np.arange(N_src), nums)
    j = []
    if same:
        for k, n in enumerate(nums):
            j += list(np.random.choice([*range(k - 1)] + [*range(k + 1, N_tar)],
                                       size=n, replace=False))
    else:
        for k, n in enumerate(nums):
            j += list(np.random.choice([*range(N_tar)],
                                       size=n, replace=False))

    return i, np.array(j)


def generate_N_connections(N_tar, N_src, N, same=False):
    """
    connect source to target with N connections per target

    return list of sources i and targets N_tar*N
    """
    if same:
        return NotImplementedError

    i = np.array([])
    j = np.repeat(range(N_tar), N)

    for k in range(N_tar):
        srcs = np.random.choice(range(N_src), size=N, replace=False)
        i = np.concatenate((i, srcs))

    i, j = i.astype(int), j.astype(int)
    assert len(i) == len(j)

    return i, j


def generate_full_connectivity(Nsrc, Ntar=0, same=True):
    if same:
        i = []
        j = []
        for k in range(Nsrc):
            i.extend([k] * (Nsrc - 1))
            targets = list(range(Nsrc))
            del targets[k]
            j.extend(targets)

        assert len(i) == len(j)
        return np.array(i), np.array(j)

    else:
        i = []
        j = []
        for k in range(Nsrc):
            i.extend([k] * Ntar)
            targets = list(range(Ntar))
            j.extend(targets)

        assert len(i) == len(j)
        return np.array(i), np.array(j)


# IP added
# util methods for distance dependent connectivity
def gaussian(x, u, s):
    # g = (2/np.sqrt(2*np.pi*s*s))*np.exp(-(x-u)*(x-u)/(2*s*s))  # gaussian used in Miner paper
    g = np.exp(-(x - u) * (x - u) / (2 * s * s))
    return g


def make_grid(grid_size: int, n_neurons: int) -> np.ndarray:
    """
    Randomly populate square-shaped grid with neurons.

    :param grid_size: grid side length in microns
    :param n_neurons: number of neurons to populate

    :return: 2d array with neurons coordinates
    """
    # todo use rand or random_integer?
    # todo handle case when neurons have the same position? Or super close position in case od rand?
    # return np.random.random_integers(0, grid_size, (n_neurons, 2))
    # return default_rng().integers(0, grid_size, endpoint=True)
    return grid_size * np.random.rand(n_neurons, 2)


def generate_dd_connectivity(tar_x, tar_y, src_x, src_y, g_halfwidth, same=True):
    # calculate connection matrix and indexes arrays
    n_tar = np.size(tar_x)
    # print('n_tar '+str(n_tar))
    n_src = np.size(src_x)
    # print('n_src '+str(n_src))
    p_ = np.zeros((n_src, n_tar))
    for i in range(n_src):
        for j in range(n_tar):
            if not same or (same and not (i == j)):
                dx = tar_x[j] - src_x[i]
                dy = tar_y[j] - src_y[i]
                p_[i, j] = gaussian(np.sqrt(dx ** 2 + dy ** 2), 0, np.array(g_halfwidth))

    conn = np.zeros((n_src, n_tar))  # connectivity matrix
    in_src = []  # list with source indexes
    in_trg = []  # list with target indexes
    nums = np.random.uniform(size=n_src*n_tar)
    # print('nums array size '+str(np.size(nums)))
    for i in range(n_src):
        for j in range(n_tar):
            if nums[i*(n_tar-1) + j] < p_[i, j]:
                in_src.append(i)
                in_trg.append(j)
                conn[i, j] = 1  # just indicate a connection, no weight set
    # print('in_src/in_trg size '+str(np.size(in_src)))
    return in_src, in_trg


def generate_dd_connectivity2(n_src):
    """
    :param n_src:
    :return:
    """
    # set parameters
    # grid = 1000  # um
    xy = 1000 * np.random.rand(n_src, 2)

    # calculate connection matrix and indexes arrays
    n_tar = n_src
    p_ = np.zeros((n_src, n_tar))
    for i in range(n_src):
        for j in range(n_tar):
            if not (i == j):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                p_[i, j] = gaussian(np.sqrt(dx ** 2 + dy ** 2), 0, 200)  # half-width 20um

    conn = np.zeros((n_src, n_tar))  # connectivity matrix
    in_src = []  # list with source indexes
    in_trg = []  # list with target indexes
    for i in range(n_src):
        for j in range(n_tar):
            if np.random.rand() < p_[i, j]:
                in_src.append(i)
                in_trg.append(j)
                conn[i, j] = 1  # just indicate a connection, no weight set
    return in_src, in_trg


def generate_dd_connectivity3(n_src, n_tar, xy):
    """

    :param xy:
    :param n_tar:
    :param n_src:
    :return:
    """
    # set parameters
    # grid = 1000  # um
    # xy = 1000 * np.random.rand(n_src, 2)

    # calculate connection matrix and indexes arrays
    # n_tar = n_src
    p_ = np.zeros((n_src, n_tar))
    for i in range(n_src):
        for j in range(n_tar):
            if not (i == j):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                p_[i, j] = gaussian(np.sqrt(dx ** 2 + dy ** 2), 0, 200)  # half-width 20um

    conn = np.zeros((n_src, n_tar))  # connectivity matrix
    in_src = []  # list with source indexes
    in_trg = []  # list with target indexes
    for i in range(n_src):
        for j in range(n_tar):
            if np.random.rand() < p_[i, j]:
                in_src.append(i)
                in_trg.append(j)
                conn[i, j] = 1  # just indicate a connection, no weight set
    return in_src, in_trg

