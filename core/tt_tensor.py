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
        if not cores:
            raise ValueError("cores не может быть пустым")
        self.cores = cores
        self.order = len(cores)
        shapes = [core.shape for core in cores]
        for k, (r_left, n, r_right) in enumerate(shapes):
            if len(shapes[k]) != 3:
                raise ValueError(f"ядро {k} должно быть 3D, получено {len(shapes[k])}D")
        for k in range(1, self.order):
            if shapes[k - 1][2] != shapes[k][0]:
                raise ValueError(
                    f"ранги не совпадают: ядро {k - 1} имеет r_right={shapes[k - 1][2]}, "
                    f"ядро {k} имеет r_left={shapes[k][0]}"
                )
        if shapes[0][0] != 1:
            raise ValueError(f"первое ядро должно иметь r_left=1, получено {shapes[0][0]}")
        if shapes[-1][2] != 1:
            raise ValueError(f"последнее ядро должно иметь r_right=1, получено {shapes[-1][2]}")
        self.shape = tuple(core.shape[1] for core in cores)
        self.ranks = (1,) + tuple(core.shape[2] for core in cores[:-1]) + (1,)


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
        if seed is not None:
            random.seed(seed)
        shape = validate_shape(shape)
        d = len(shape)
        if isinstance(ranks, (tuple, list)):
            if len(ranks) == d + 1:
                if ranks[0] != 1 or ranks[-1] != 1:
                    raise ValueError("граничные ранги должны быть равны 1")
                ranks = list(ranks)
            elif len(ranks) == d - 1:
                ranks = [1] + list(ranks) + [1]
            else:
                raise ValueError(f"неверная длина ranks: {len(ranks)}, ожидается {d - 1} или {d + 1}")
        else:
            raise ValueError("ranks должен быть tuple или list")
        cores = []
        for k in range(d):
            r_left = ranks[k]
            n_k = shape[k]
            r_right = ranks[k + 1]
            size = r_left * n_k * r_right
            data = [random.uniform(-1.0, 1.0) for _ in range(size)]
            cores.append(DenseTensor((r_left, n_k, r_right), data=data))
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
        if len(indices) != self.order:
            raise ValueError(f"ожидается {self.order} индексов, получено {len(indices)}")
        for idx, dim in zip(indices, self.shape):
            if idx < 0 or idx >= dim:
                raise IndexError(f"индекс {idx} вне диапазона [0, {dim})")
        result = 1.0
        for k in range(self.order):
            core = self.cores[k]
            r_left = core.shape[0]
            r_right = core.shape[2]
            if k == 0:
                vec = [core[0, indices[0], j] for j in range(r_right)]
                result = vec
            elif k == self.order - 1:
                s = 0.0
                for i in range(r_left):
                    s += result[i] * core[i, indices[k], 0]
                result = s
            else:
                new_vec = []
                for j in range(r_right):
                    s = 0.0
                    for i in range(r_left):
                        s += result[i] * core[i, indices[k], j]
                    new_vec.append(s)
                result = new_vec
        return float(result)

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result = DenseTensor.ones(self.shape)
        for flat_idx in range(result.size):
            multi = []
            remaining = flat_idx
            for dim in reversed(self.shape):
                multi.append(remaining % dim)
                remaining //= dim
            multi = tuple(reversed(multi))
            val = 1.0
            for k in range(self.order):
                core = self.cores[k]
                if k == 0:
                    vec = [core[0, multi[0], j] for j in range(core.shape[2])]
                    val = vec
                elif k == self.order - 1:
                    s = 0.0
                    for i in range(core.shape[0]):
                        s += val[i] * core[i, multi[k], 0]
                    val = s
                else:
                    new_vec = []
                    for j in range(core.shape[2]):
                        s = 0.0
                        for i in range(core.shape[0]):
                            s += val[i] * core[i, multi[k], j]
                        new_vec.append(s)
                    val = new_vec
            result.data[flat_idx] = float(val)
        return result

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
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        tt_size = self.total_storage()
        if tt_size == 0:
            return float('inf')
        return full_size / tt_size

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
        lines = [
            f"TTTensor(order={self.order}, shape={self.shape}, ranks={self.ranks})",
            f"  cores shapes: {self.core_sizes()}",
            f"  total storage: {self.total_storage()} elements",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()