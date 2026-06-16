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
    d = tensor.ndim
    if d == 1:
        core = tensor.reshape((1, tensor.shape[0], 1))
        return TTTensor([core])
    norm_tensor = tensor.norm()
    if norm_tensor < 1e-30:
        delta = 0.0
    else:
        delta = eps * norm_tensor / math.sqrt(d - 1)
    C = tensor.copy()
    cores = []
    r_left = 1
    for k in range(d - 1):
        mat = C.left_unfolding(k)
        U, S, Vt = backend.svd(mat)
        new_r = _compute_truncated_rank(S, delta, max_rank)
        if new_r == 0:
            new_r = 1
        U_trunc = _truncate_columns(U, new_r, backend)
        core_data = []
        for i in range(r_left):
            for idx in range(tensor.shape[k]):
                for j in range(new_r):
                    core_data.append(U_trunc[i * tensor.shape[k] + idx, j])
        cores.append(DenseTensor((r_left, tensor.shape[k], new_r), data=core_data))
        S_trunc = _truncate_vector(S, new_r, backend)
        Vt_trunc = _truncate_rows(Vt, new_r, backend)
        SV_data = []
        for i in range(new_r):
            for j in range(Vt_trunc.shape[1]):
                SV_data.append(S_trunc[i] * Vt_trunc[i, j])
        remaining_shape = tensor.shape[k + 1:]
        new_shape = (new_r,) + remaining_shape
        C = DenseTensor(new_shape, data=SV_data)
        r_left = new_r
    last_data = []
    for i in range(r_left):
        for idx in range(tensor.shape[-1]):
            last_data.append(C[i, idx])
    cores.append(DenseTensor((r_left, tensor.shape[-1], 1), data=last_data))
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
        raise ValueError(f"S должен быть 1D, получен {S.ndim}D")
    if S.size == 0:
        return 0
    max_s = max(S.data)
    threshold = max(1e-12, 1e-8 * max_s)
    rank = 0
    for val in S.data:
        if val > threshold:
            rank += 1
        else:
            break
    if delta > 0:
        for r in range(rank, 0, -1):
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