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
    if max_rank is not None:
        if not isinstance(max_rank, int) or max_rank <= 0:
            raise ValueError("max_rank must be positive integer or None")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if tt.order == 1:
        return tt.copy()
    rounded = _right_orthogonalize_for_round(tt, backend)
    cores = [core.copy() for core in rounded.cores]
    first_norm = backend.norm(cores[0])
    if first_norm > 1e-30:
        delta = eps * first_norm / math.sqrt(tt.order - 1)
    else:
        delta = 0.0
    for k in range(tt.order - 1):
        core = cores[k]
        r_left, n, r_right = core.shape
        matrix = backend.reshape(core, (r_left * n, r_right))
        U, S, Vt = _safe_svd(matrix, backend)
        rank = _compute_rank(S, delta, max_rank)
        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        cores[k] = backend.reshape(U_trunc, (r_left, n, rank))
        rest = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)
        next_core = cores[k + 1]
        _, next_n, next_r = next_core.shape
        new_next = backend.zeros((rank, next_n, next_r))
        for a in range(rank):
            for i in range(next_n):
                for b in range(next_r):
                    val = 0.0
                    for c in range(r_right):
                        val = val + rest[(a, c)] * next_core[(c, i, b)]
                    new_next[(a, i, b)] = val
        cores[k + 1] = new_next
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _right_orthogonalize_for_round(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Внутренняя правая ортогонализация для TT-round.
    """
    cores = [core.copy() for core in tt.cores]
    for k in range(tt.order - 1, 0, -1):
        core = cores[k]
        r_left, n, r_right = core.shape
        matrix = backend.reshape(core, (r_left, n * r_right))
        U, S, Vt = _safe_svd(matrix, backend)
        rank = S.shape[0]
        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        cores[k] = backend.reshape(Vt_trunc, (rank, n, r_right))
        left_part = _multiply_columns_by_diag(U_trunc, S_trunc, backend)
        prev_core = cores[k - 1]
        prev_r_left, prev_n, _ = prev_core.shape
        new_prev = backend.zeros((prev_r_left, prev_n, rank))
        for a in range(prev_r_left):
            for i in range(prev_n):
                for b in range(rank):
                    val = 0.0
                    for c in range(r_left):
                        val = val + prev_core[(a, i, c)] * left_part[(c, b)]
                    new_prev[(a, i, b)] = val
        cores[k - 1] = new_prev
    return TTTensor(cores)


def _safe_svd(matrix: DenseTensor, backend: BackendInterface) -> tuple[DenseTensor, DenseTensor, DenseTensor]:
    """
    SVD для любых матриц.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    rows, cols = matrix.shape
    if rows >= cols:
        return backend.svd(matrix, full_matrices=False)
    matrix_t = _transpose_matrix(matrix, backend)
    U_t, S, Vt_t = backend.svd(matrix_t, full_matrices=False)
    U = _transpose_matrix(Vt_t, backend)
    Vt = _transpose_matrix(U_t, backend)
    return U, S, Vt


def _transpose_matrix(matrix: DenseTensor, backend: BackendInterface) -> DenseTensor:
    """Возвращает транспонированную матрицу."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    rows, cols = matrix.shape
    result = backend.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            result[(j, i)] = matrix[(i, j)]
    return result


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
        raise ValueError("S must be vector")
    count = S.shape[0]
    if count == 0:
        return 1
    border = max(1e-12, 1e-8 * abs(S[0]))
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
        raise ValueError("matrix must be 2-dimensional")
    rows, cols = matrix.shape
    if rank < 1 or rank > cols:
        raise ValueError("wrong rank")
    result = backend.zeros((rows, rank))
    for i in range(rows):
        for j in range(rank):
            result[(i, j)] = matrix[(i, j)]
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
            result[(i, j)] = matrix[(i, j)]
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
        result[(i,)] = vector[(i,)]
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
        diag_val = diag_vec[(i,)]
        for j in range(cols):
            result[(i, j)] = diag_val * matrix[(i, j)]
    return result


def _multiply_columns_by_diag(
        matrix: DenseTensor,
        diag_vec: DenseTensor,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec must be vector")

    rows, cols = matrix.shape
    if cols != diag_vec.shape[0]:
        raise ValueError("wrong shapes")

    result = backend.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[(i, j)] = matrix[(i, j)] * diag_vec[(j,)]
    return result

