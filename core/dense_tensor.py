# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)
        if data is None:
            self.data = [fill] * self.size
        else:
            if len(data) != self.size:
                raise ValueError(f"длина данных {len(data)} не совпадает с размером {self.size}")
            self.data = list(data)

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        if seed is not None:
            random.seed(seed)
        shape = validate_shape(shape)
        size = compute_size(shape)
        if integer:
            data = [random.randint(low, high) for _ in range(size)]
        else:
            data = [random.uniform(low, high) for _ in range(size)]
        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """


        def get_shape(lst):
            if not isinstance(lst, list):
                return ()
            if not lst:
                return (0,)
            shape = [len(lst)]
            sub_shape = get_shape(lst[0])
            shape.extend(sub_shape)
            return tuple(shape)


        def flatten(lst):
            if not isinstance(lst, list):
                return [float(lst)]
            result = []
            for item in lst:
                result.extend(flatten(item))
            return result
        shape = get_shape(nested)
        data = flatten(nested)
        if compute_size(shape) != len(data):
            raise ValueError("некорректная вложенная структура")
        return DenseTensor(shape, data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int):
            if multi_index < 0 or multi_index >= self.size:
                raise IndexError(f"индекс {multi_index} вне диапазона [0, {self.size})")
            return flat_to_multi_index(multi_index, self.shape)
        if len(multi_index) != self.ndim:
            raise IndexError(f"ожидается {self.ndim} индексов, получено {len(multi_index)}")
        for idx, dim in zip(multi_index, self.shape):
            if idx < 0 or idx >= dim:
                raise IndexError(f"индекс {idx} вне диапазона [0, {dim})")
        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        idx = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx, self.strides)
        return self.data[flat_idx]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        idx = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx, self.strides)
        self.data[flat_idx] = float(value)

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        new_shape = validate_shape(new_shape)
        if compute_size(new_shape) != self.size:
            raise ValueError(f"невозможно reshape: {self.size} != {compute_size(new_shape)}")
        return DenseTensor(new_shape, data=self.data.copy())

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"mode должен быть в [0, {self.ndim})")
        rows = self.shape[mode]
        cols = self.size // rows
        result = DenseTensor.zeros((rows, cols))
        for flat_idx in range(self.size):
            multi = flat_to_multi_index(flat_idx, self.shape)
            row = multi[mode]
            col = 0
            for k in range(self.ndim):
                if k != mode:
                    col = col * self.shape[k] + multi[k]
            result[row, col] = self.data[flat_idx]
        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"k должен быть в [0, {self.ndim - 1})")
        rows = 1
        for i in range(k + 1):
            rows *= self.shape[i]
        cols = self.size // rows
        result = DenseTensor.zeros((rows, cols))
        for flat_idx in range(self.size):
            multi = flat_to_multi_index(flat_idx, self.shape)
            row = 0
            for i in range(k + 1):
                row = row * self.shape[i] + multi[i]
            col = 0
            for i in range(k + 1, self.ndim):
                col = col * self.shape[i] + multi[i]
            result[row, col] = self.data[flat_idx]
        return result

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data.copy())

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        data = [a + b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        data = [a - b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, data=data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self.__mul__(-1.0)

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if self.shape != other.shape:
            return False
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        if self.ndim == 1:
            return self.data.copy()
        result = []
        for flat_idx in range(self.size):
            multi = flat_to_multi_index(flat_idx, self.shape)
            node = result
            for i in range(self.ndim - 1):
                idx = multi[i]
                while len(node) <= idx:
                    node.append([])
                node = node[idx]
            node.append(self.data[flat_idx])
        return result

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.data[:10]}{'...' if self.size > 10 else ''})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()