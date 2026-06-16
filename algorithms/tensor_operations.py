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
        raise ValueError("TT tensors must have same shape")
    cores = []
    d = tt1.order
    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape
        if k == 0:
            new_core = backend.zeros((1, n, r1_right + r2_right))
            for i in range(n):
                for a in range(r1_right):
                    new_core[(0, i, a)] = core1[(0, i, a)]
                for b in range(r2_right):
                    new_core[(0, i, r1_right + b)] = core2[(0, i, b)]
        elif k == d - 1:
            new_core = backend.zeros((r1_left + r2_left, n, 1))
            for i in range(n):
                for a in range(r1_left):
                    new_core[(a, i, 0)] = core1[(a, i, 0)]
                for b in range(r2_left):
                    new_core[(r1_left + b, i, 0)] = core2[(b, i, 0)]
        else:
            new_core = backend.zeros((r1_left + r2_left, n, r1_right + r2_right))
            for i in range(n):
                for a in range(r1_left):
                    for b in range(r1_right):
                        new_core[(a, i, b)] = core1[(a, i, b)]
                for a in range(r2_left):
                    for b in range(r2_right):
                        new_core[(r1_left + a, i, r1_right + b)] = core2[(a, i, b)]
        cores.append(new_core)
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
    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be number")
    cores = [core.copy() for core in tt.cores]
    cores[0] = backend.scale(cores[0], alpha)
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
        raise ValueError("TT tensors must have same shape")
    cores = []
    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape
        new_core = backend.zeros((r1_left * r2_left, n, r1_right * r2_right))
        for i in range(n):
            for a1 in range(r1_left):
                for a2 in range(r2_left):
                    left_idx = a1 * r2_left + a2
                    for b1 in range(r1_right):
                        for b2 in range(r2_right):
                            right_idx = b1 * r2_right + b2
                            new_core[(left_idx, i, right_idx)] = (
                                    core1[(a1, i, b1)] * core2[(a2, i, b2)]
                            )
        cores.append(new_core)

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
        raise ValueError("TT tensors must have same shape")
    z = backend.ones((1, 1))
    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape
        new_z = backend.zeros((r1_right, r2_right))
        for i in range(n):
            for a1 in range(r1_left):
                for a2 in range(r2_left):
                    z_val = z[(a1, a2)]
                    for b1 in range(r1_right):
                        for b2 in range(r2_right):
                            new_z[(b1, b2)] = new_z[(b1, b2)] + (
                                    core1[(a1, i, b1)] * z_val * core2[(a2, i, b2)]
                            )
        z = new_z
    return z[(0, 0)]


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
    val = tt_dot(tt, tt, backend)
    if val < 0 and abs(val) < 1e-10:
        val = 0.0
    return math.sqrt(val)


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
    diff = tt_add(tt1, tt_scalar_mul(tt2, -1, backend), backend)
    return tt_norm(diff, backend)


