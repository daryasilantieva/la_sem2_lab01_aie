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
    d = len(cores)
    for k in range(d - 1):
        r_left, n_k, r_right = cores[k].shape
        mat_data = []
        for i in range(r_left):
            for idx in range(n_k):
                for j in range(r_right):
                    mat_data.append(cores[k][i, idx, j])
        mat = DenseTensor((r_left * n_k, r_right), data=mat_data)
        Q, R = backend.qr(mat)
        new_r = Q.shape[1]
        q_data = []
        for i in range(r_left):
            for idx in range(n_k):
                for j in range(new_r):
                    q_data.append(Q[i * n_k + idx, j])
        cores[k] = DenseTensor((r_left, n_k, new_r), data=q_data)
        next_r_left, next_n, next_r_right = cores[k + 1].shape
        new_next_data = []
        for i in range(new_r):
            for idx in range(next_n):
                for j in range(next_r_right):
                    s = 0.0
                    for p in range(r_right):
                        s += R[i, p] * cores[k + 1][p, idx, j]
                    new_next_data.append(s)
        cores[k + 1] = DenseTensor((new_r, next_n, next_r_right), data=new_next_data)
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    d = len(cores)
    for k in range(d - 1, 0, -1):
        r_left, n_k, r_right = cores[k].shape
        mat_data = []
        for i in range(r_left):
            for idx in range(n_k):
                for j in range(r_right):
                    mat_data.append(cores[k][i, idx, j])
        mat = DenseTensor((r_left, n_k * r_right), data=mat_data)
        mat_t = backend.transpose(mat)
        Q_t, R_t = backend.qr(mat_t)
        new_r = Q_t.shape[1]
        q_data = []
        for i in range(new_r):
            for idx in range(n_k):
                for j in range(r_right):
                    q_data.append(Q_t[idx * n_k + j, i])
        cores[k] = DenseTensor((new_r, n_k, r_right), data=q_data)
        prev_r_left, prev_n, prev_r_right = cores[k - 1].shape
        r_t = backend.transpose(R_t)
        new_prev_data = []
        for i in range(prev_r_left):
            for idx in range(prev_n):
                for j in range(new_r):
                    s = 0.0
                    for p in range(prev_r_right):
                        s += cores[k - 1][i, idx, p] * r_t[p, j]
                    new_prev_data.append(s)
        cores[k - 1] = DenseTensor((prev_r_left, prev_n, new_r), data=new_prev_data)
    return TTTensor(cores)


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
        raise ValueError(f"S должен быть 1D, получен {S.ndim}D")
    if S.size == 0:
        return 0
    max_s = max(S.data)
    threshold = max(abs_tol, rel_tol * max_s)
    rank = 0
    for val in S.data:
        if val > threshold:
            rank += 1
        else:
            break
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
        rank:     длина диагонального вектора
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
        raise ValueError(f"matrix должен быть 2D, получен {matrix.ndim}D")
    if diag_vec.ndim != 1:
        raise ValueError(f"diag_vec должен быть 1D, получен {diag_vec.ndim}D")
    m, n = matrix.shape
    if n != diag_vec.size:
        raise ValueError(f"matrix.shape[1]={n} != diag_vec.size={diag_vec.size}")
    data = []
    for i in range(m):
        for j in range(n):
            data.append(matrix[i, j] * diag_vec[j])
    return DenseTensor((m, n), data=data)