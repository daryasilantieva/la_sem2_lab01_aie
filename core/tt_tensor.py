# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations
import random
from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not isinstance(cores, list):
            raise TypeError("cores must be a list")
        if len(cores) == 0:
            raise ValueError("cores list must not be empty")
        for core in cores:
            if not isinstance(core, DenseTensor):
                raise TypeError("all cores must be DenseTensor")
            if core.ndim != 3:
                raise ValueError("each core must be 3-dimensional")
        if cores[0].shape[0] != 1:
            raise ValueError("first TT-rank must be 1")
        if cores[-1].shape[2] != 1:
            raise ValueError("last TT-rank must be 1")
        for i in range(len(cores) - 1):
            if cores[i].shape[2] != cores[i + 1].shape[0]:
                raise ValueError("neighbor TT-ranks do not match")
        self.cores = [core.copy() for core in cores]
        self.order = len(cores)
        self.shape = tuple(core.shape[1] for core in cores)
        ranks = [cores[0].shape[0]]
        for core in cores:
            ranks.append(core.shape[2])
        self.ranks = tuple(ranks)

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        shape = validate_shape(shape)
        order = len(shape)
        if not isinstance(ranks, (tuple, list)):
            raise TypeError("ranks must be tuple or list")
        ranks = tuple(ranks)
        if len(ranks) == order - 1:
            ranks = (1,) + ranks + (1,)
        elif len(ranks) != order + 1:
            raise ValueError("wrong number of TT-ranks")
        for rank in ranks:
            if not isinstance(rank, int) or rank <= 0:
                raise ValueError("all ranks must be positive integers")
        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("first and last TT-ranks must be 1")
        cores = []
        for k in range(order):
            core_seed = seed + k if seed is not None else None
            core_shape = (ranks[k], shape[k], ranks[k + 1])
            cores.append(DenseTensor.random(core_shape, low=-5, high=5, integer=False, seed=core_seed))
        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if not isinstance(indices, (tuple, list)):
            raise TypeError("indices must be tuple or list")
        if len(indices) != self.order:
            raise ValueError("wrong number of indices")
        for idx, dim in zip(indices, self.shape):
            if not isinstance(idx, int):
                raise TypeError("all indices must be integers")
            if idx < 0 or idx >= dim:
                raise IndexError("index out of range")
        current = [1.0]
        for k in range(self.order):
            core = self.cores[k]
            mode_idx = indices[k]
            next_vec = [0.0 for _ in range(core.shape[2])]
            for left in range(core.shape[0]):
                for right in range(core.shape[2]):
                    next_vec[right] = next_vec[right] + current[left] * core[(left, mode_idx, right)]
            current = next_vec
        return current[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result = DenseTensor.zeros(self.shape)

        def traverse(current_indices, mode):
            if mode == self.order:
                result[tuple(current_indices)] = self.get_element(current_indices)
                return
            for i in range(self.shape[mode]):
                current_indices.append(i)
                traverse(current_indices, mode + 1)
                current_indices.pop()

        traverse([], 0)
        return result

    def to_dense(self):
        return self.full()

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        total = 0
        for core in self.cores:
            total = total + core.size
        return total

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        return compute_size(self.shape) / self.total_storage()

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        return (
            "TTTensor(\n"
            f"  order={self.order},\n"
            f"  shape={self.shape},\n"
            f"  ranks={self.ranks},\n"
            f"  core_sizes={self.core_sizes()},\n"
            f"  total_storage={self.total_storage()}\n"
            ")"
        )

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()

