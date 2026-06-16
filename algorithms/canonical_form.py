# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    for k in range(tt.order - 1):
        core = cores[k]
        r_left, n, r_right = core.shape
        matrix = backend.reshape(core, (r_left * n, r_right))
        Q, R = backend.qr(matrix)
        new_rank = Q.shape[1]
        cores[k] = backend.reshape(Q, (r_left, n, new_rank))
        next_core = cores[k + 1]
        _, next_n, next_r = next_core.shape
        new_next = backend.zeros((new_rank, next_n, next_r))
        for a in range(new_rank):
            for i in range(next_n):
                for b in range(next_r):
                    val = 0.0
                    for c in range(r_right):
                        val = val + R[(a, c)] * next_core[(c, i, b)]
                    new_next[(a, i, b)] = val
        cores[k + 1] = new_next
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    for k in range(tt.order - 1, 0, -1):
        core = cores[k]
        r_left, n, r_right = core.shape
        matrix = backend.reshape(core, (r_left, n * r_right))
        if n * r_right >= r_left:
            matrix_t = _transpose_matrix(matrix, backend)
            Q_t, R_t = backend.qr(matrix_t)
            rank = Q_t.shape[1]
            Q = _transpose_matrix(Q_t, backend)
            R = _transpose_matrix(R_t, backend)
        else:
            U, S, Vt = backend.svd(matrix)
            rank = S.shape[0]
            Q = Vt
            R = _multiply_columns_by_diag(U, S, backend)
        cores[k] = backend.reshape(Q, (rank, n, r_right))
        prev_core = cores[k - 1]
        prev_r_left, prev_n, _ = prev_core.shape
        new_prev = backend.zeros((prev_r_left, prev_n, rank))
        for a in range(prev_r_left):
            for i in range(prev_n):
                for b in range(rank):
                    val = 0.0
                    for c in range(r_left):
                        val = val + prev_core[(a, i, c)] * R[(c, b)]
                    new_prev[(a, i, b)] = val
        cores[k - 1] = new_prev
    return TTTensor(cores)


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


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.ndim != 1:
        raise ValueError("S must be vector")
    if S.shape[0] == 0:
        return 0
    max_s = 0.0
    for i in range(S.shape[0]):
        if abs(S[i]) > max_s:
            max_s = abs(S[i])
    border = max(abs_tol, rel_tol * max_s)
    cnt = 0
    for i in range(S.shape[0]):
        if abs(S[i]) > border:
            cnt = cnt + 1
    return max(1, cnt)


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
        raise ValueError("vector must be 1")
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
        rank:     длина диагонального вектора
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

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
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


