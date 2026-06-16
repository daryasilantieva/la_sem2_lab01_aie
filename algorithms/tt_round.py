# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    tt_orth = right_canonicalize(tt, backend)
    cores = [core.copy() for core in tt_orth.cores]
    d = len(cores)
    norm_g1 = 0.0
    r_left, n, r_right = cores[0].shape
    for i in range(r_left):
        for idx in range(n):
            for j in range(r_right):
                norm_g1 += cores[0][i, idx, j] ** 2
    norm_g1 = math.sqrt(norm_g1)
    if d > 1 and norm_g1 > 1e-30:
        delta = eps * norm_g1 / math.sqrt(d - 1)
    else:
        delta = 0.0
    for k in range(d - 1):
        r_left, n_k, r_right = cores[k].shape

        mat_data = []
        for i in range(r_left):
            for idx in range(n_k):
                for j in range(r_right):
                    mat_data.append(cores[k][i, idx, j])
        mat = DenseTensor((r_left * n_k, r_right), data=mat_data)
        U, S, Vt = backend.svd(mat)
        new_r = _compute_rank(S, delta, max_rank)
        if new_r == 0:
            new_r = 1
        U_trunc = _truncate_columns(U, new_r, backend)
        core_data = []
        for i in range(r_left):
            for idx in range(n_k):
                for j in range(new_r):
                    core_data.append(U_trunc[i * n_k + idx, j])
        cores[k] = DenseTensor((r_left, n_k, new_r), data=core_data)
        S_trunc = [S[i] for i in range(new_r)]
        Vt_trunc = _truncate_rows(Vt, new_r, backend)
        SV_data = []
        for i in range(new_r):
            for j in range(Vt_trunc.shape[1]):
                SV_data.append(S_trunc[i] * Vt_trunc[i, j])
        SV = DenseTensor((new_r, Vt_trunc.shape[1]), data=SV_data)
        next_r_left, next_n, next_r_right = cores[k + 1].shape
        new_next_data = []
        for i in range(new_r):
            for idx in range(next_n):
                for j in range(next_r_right):
                    s = 0.0
                    for p in range(r_right):
                        s += SV[i, p] * cores[k + 1][p, idx, j]
                    new_next_data.append(s)
        cores[k + 1] = DenseTensor((new_r, next_n, next_r_right), data=new_next_data)
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError(f"S должен быть 1D, получен {S.ndim}D")
    if S.size == 0:
        return 0
    rank = len(S.data)
    for r in range(1, len(S.data) + 1):
        discarded = sum(s * s for s in S.data[r:])
        if discarded <= delta * delta:
            rank = r
            break
    if max_rank is not None and rank > max_rank:
        rank = max_rank
    return max(1, rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix должен быть 2D, получен {matrix.ndim}D")
    m, n = matrix.shape
    if rank > n:
        raise ValueError(f"rank={rank} > n={n}")
    data = []
    for i in range(m):
        for j in range(rank):
            data.append(matrix[i, j])
    return DenseTensor((m, rank), data=data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix должен быть 2D, получен {matrix.ndim}D")
    k, n = matrix.shape
    if rank > k:
        raise ValueError(f"rank={rank} > k={k}")
    data = []
    for i in range(rank):
        for j in range(n):
            data.append(matrix[i, j])
    return DenseTensor((rank, n), data=data)


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if vector.ndim != 1:
        raise ValueError(f"vector должен быть 1D, получен {vector.ndim}D")
    if rank > vector.size:
        raise ValueError(f"rank={rank} > size={vector.size}")
    return DenseTensor((rank,), data=vector.data[:rank])


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim != 1:
        raise ValueError(f"diag_vec должен быть 1D, получен {diag_vec.ndim}D")
    if matrix.ndim != 2:
        raise ValueError(f"matrix должен быть 2D, получен {matrix.ndim}D")
    if diag_vec.size != rank:
        raise ValueError(f"diag_vec.size={diag_vec.size} != rank={rank}")
    if matrix.shape[0] != rank:
        raise ValueError(f"matrix.shape[0]={matrix.shape[0]} != rank={rank}")
    m, n = matrix.shape
    data = []
    for i in range(m):
        for j in range(n):
            data.append(diag_vec[i] * matrix[i, j])
    return DenseTensor((m, n), data=data)