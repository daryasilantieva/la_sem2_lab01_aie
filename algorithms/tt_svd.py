# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if not hasattr(tensor, "shape") or not hasattr(tensor, "ndim") or not hasattr(tensor, "data"):
        raise TypeError("tensor must have shape, ndim and data")
    if max_rank is not None:
        if not isinstance(max_rank, int) or max_rank <= 0:
            raise ValueError("max_rank must be positive integer or None")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    shape = tuple(tensor.shape)
    d = tensor.ndim
    dense = DenseTensor(shape, list(tensor.data))
    if d == 1:
        core = backend.reshape(dense, (1, shape[0], 1))
        return TTTensor([core])
    tensor_norm = backend.norm(dense)
    if tensor_norm > 1e-30:
        delta = eps * tensor_norm / math.sqrt(d - 1)
    else:
        delta = 0.0
    cores = []
    C = dense.copy()
    r_prev = 1
    for k in range(d - 1):
        n_k = shape[k]
        rows = r_prev * n_k
        cols = 1
        for j in range(k + 1, d):
            cols = cols * shape[j]
        C = backend.reshape(C, (rows, cols))
        U, S, Vt = backend.svd(C, full_matrices=False)
        rank = _compute_truncated_rank(S, delta, max_rank)
        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        core = backend.reshape(U_trunc, (r_prev, n_k, rank))
        cores.append(core)
        C = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)
        r_prev = rank
    last_core = backend.reshape(C, (r_prev, shape[-1], 1))
    cores.append(last_core)
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError("S must be vector")
    count = S.shape[0]
    if count == 0:
        return 1
    first = abs(S[0])
    border = max(1e-12, 1e-8 * first)
    numerical_rank = 0
    for i in range(count):
        if abs(S[i]) > border:
            numerical_rank = i + 1
    if numerical_rank == 0:
        numerical_rank = 1
    rank = numerical_rank
    if delta > 0:
        tail_sum = 0.0
        for i in range(numerical_rank - 1, 0, -1):
            tail_sum = tail_sum + S[i] * S[i]
            if tail_sum <= delta * delta:
                rank = i
            else:
                break
    if max_rank is not None:
        rank = min(rank, max_rank)
    rank = max(1, rank)
    return rank


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    rows, cols = matrix.shape
    if rank < 1 or rank > cols:
        raise ValueError("wrong rank")
    result = backend.zeros((rows, rank))
    for i in range(rows):
        for j in range(rank):
            result[(i, j)] = backend.get_element(matrix, (i, j))
    return result


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
        raise ValueError("matrix must be 2-dimensional")
    rows, cols = matrix.shape
    if rank < 1 or rank > rows:
        raise ValueError("wrong rank")
    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            result[(i, j)] = backend.get_element(matrix, (i, j))
    return result


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
        raise ValueError("vector must be 1-dimensional")
    if rank < 1 or rank > vector.shape[0]:
        raise ValueError("wrong rank")
    result = backend.zeros((rank,))
    for i in range(rank):
        result[(i,)] = backend.get_element(vector, (i,))
    return result


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
        raise ValueError("diag_vec must be vector")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if diag_vec.shape[0] != rank or matrix.shape[0] != rank:
        raise ValueError("wrong shapes")
    cols = matrix.shape[1]
    result = backend.zeros((rank, cols))
    for i in range(rank):
        diag_val = backend.get_element(diag_vec, (i,))
        for j in range(cols):
            val = diag_val * backend.get_element(matrix, (i, j))
            backend.set_element(result, (i, j), val)
    return result

