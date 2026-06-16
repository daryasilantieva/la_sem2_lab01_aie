# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"формы не совпадают: {tt1.shape} != {tt2.shape}")
    cores = []
    d = tt1.order
    core1_data = []
    r1_left, n1, r1_right = tt1.cores[0].shape
    r2_left, n2, r2_right = tt2.cores[0].shape
    new_r_right = r1_right + r2_right
    for i in range(1):
        for idx in range(n1):
            for j in range(new_r_right):
                if j < r1_right:
                    core1_data.append(tt1.cores[0][i, idx, j])
                else:
                    core1_data.append(tt2.cores[0][i, idx, j - r1_right])
    cores.append(DenseTensor((1, n1, new_r_right), data=core1_data))
    for k in range(1, d - 1):
        r1_left, n_k, r1_right = tt1.cores[k].shape
        r2_left, n2, r2_right = tt2.cores[k].shape
        new_r_left = r1_left + r2_left
        new_r_right = r1_right + r2_right
        core_data = []
        for i in range(new_r_left):
            for idx in range(n_k):
                for j in range(new_r_right):
                    if i < r1_left and j < r1_right:
                        core_data.append(tt1.cores[k][i, idx, j])
                    elif i >= r1_left and j >= r1_right:
                        core_data.append(tt2.cores[k][i - r1_left, idx, j - r1_right])
                    else:
                        core_data.append(0.0)
        cores.append(DenseTensor((new_r_left, n_k, new_r_right), data=core_data))
    r1_left, n_d, r1_right = tt1.cores[-1].shape
    r2_left, n2, r2_right = tt2.cores[-1].shape
    new_r_left = r1_left + r2_left
    core_data = []
    for i in range(new_r_left):
        for idx in range(n_d):
            for j in range(1):
                if i < r1_left:
                    core_data.append(tt1.cores[-1][i, idx, 0])
                else:
                    core_data.append(tt2.cores[-1][i - r1_left, idx, 0])
    cores.append(DenseTensor((new_r_left, n_d, 1), data=core_data))
    return TTTensor(cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    r_left, n, r_right = cores[0].shape
    new_data = []
    for i in range(r_left):
        for idx in range(n):
            for j in range(r_right):
                new_data.append(cores[0][i, idx, j] * alpha)
    cores[0] = DenseTensor((r_left, n, r_right), data=new_data)
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"формы не совпадают: {tt1.shape} != {tt2.shape}")
    cores = []
    d = tt1.order
    for k in range(d):
        r1_left, n_k, r1_right = tt1.cores[k].shape
        r2_left, n2, r2_right = tt2.cores[k].shape
        new_r_left = r1_left * r2_left
        new_r_right = r1_right * r2_right
        core_data = []
        for i1 in range(r1_left):
            for i2 in range(r2_left):
                for idx in range(n_k):
                    for j1 in range(r1_right):
                        for j2 in range(r2_right):
                            core_data.append(tt1.cores[k][i1, idx, j1] * tt2.cores[k][i2, idx, j2])
        cores.append(DenseTensor((new_r_left, n_k, new_r_right), data=core_data))
    return TTTensor(cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"формы не совпадают: {tt1.shape} != {tt2.shape}")
    d = tt1.order
    r1_left, n1, r1_right = tt1.cores[0].shape
    r2_left, n2, r2_right = tt2.cores[0].shape
    Z = DenseTensor.zeros((r1_right, r2_right))
    for idx in range(n1):
        for i in range(r1_right):
            for j in range(r2_right):
                Z[i, j] += tt1.cores[0][0, idx, i] * tt2.cores[0][0, idx, j]
    for k in range(1, d):
        r1_left, n_k, r1_right = tt1.cores[k].shape
        r2_left, n2, r2_right = tt2.cores[k].shape
        new_Z = DenseTensor.zeros((r1_right, r2_right))
        for idx in range(n_k):
            for i in range(r1_left):
                for j in range(r2_left):
                    z_val = Z[i, j]
                    if z_val != 0.0:
                        for a in range(r1_right):
                            for b in range(r2_right):
                                new_Z[a, b] += tt1.cores[k][i, idx, a] * z_val * tt2.cores[k][j, idx, b]
        Z = new_Z
    return Z[0, 0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    return math.sqrt(tt_dot(tt, tt, backend))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    norm_sq = tt_dot(tt1, tt1, backend) + tt_dot(tt2, tt2, backend) - 2 * tt_dot(tt1, tt2, backend)
    return math.sqrt(max(0.0, norm_sq))