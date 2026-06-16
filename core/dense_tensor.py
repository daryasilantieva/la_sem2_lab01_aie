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
            self.data = [float(fill) for _ in range(self.size)]
        else:
            if len(data) != self.size:
                raise ValueError("data length does not match tensor size")
            self.data = [float(item) for item in data]

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
        shape = validate_shape(shape)
        size = compute_size(shape)
        if seed is not None:
            random.seed(seed)
        data = []
        for _ in range(size):
            if integer:
                val = float(random.randint(low, high))
            else:
                val = float(random.uniform(low, high))
            data.append(val)
        return DenseTensor(shape, data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """

        def get_shape(obj):
            if not isinstance(obj, (list, tuple)):
                return ()
            if len(obj) == 0:
                raise ValueError("nested list must not be empty")
            first = get_shape(obj[0])
            for item in obj:
                if get_shape(item) != first:
                    raise ValueError("nested list must have regular shape")
            return (len(obj),) + first

        def flatten(obj):
            if not isinstance(obj, (list, tuple)):
                return [float(obj)]
            res = []
            for item in obj:
                res.extend(flatten(item))
            return res
        shape = get_shape(nested)
        data = flatten(nested)
        return DenseTensor(shape, data)

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
            multi_index = (multi_index,)
        if not isinstance(multi_index, tuple):
            raise TypeError("index must be int or tuple")
        if len(multi_index) != self.ndim:
            raise ValueError("wrong number of indices")
        for idx, dim in zip(multi_index, self.shape):
            if not isinstance(idx, int):
                raise TypeError("all indices must be integers")
            if idx < 0 or idx >= dim:
                raise IndexError("index out of range")
        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        multi_index = self._validate_index(multi_index)
        flat = multi_index_to_flat(multi_index, self.strides)
        return self.data[flat]

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
        multi_index = self._validate_index(multi_index)
        flat = multi_index_to_flat(multi_index, self.strides)
        self.data[flat] = float(value)

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
            raise ValueError("new shape must have the same size")
        return DenseTensor(new_shape, self.data.copy())

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if not isinstance(mode, int):
            raise TypeError("mode must be integer")
        if mode < 0 or mode >= self.ndim:
            raise ValueError("mode out of range")
        rows = self.shape[mode]
        cols = self.size // rows
        result = DenseTensor.zeros((rows, cols))
        other_modes = [i for i in range(self.ndim) if i != mode]
        other_shape = [self.shape[i] for i in other_modes]
        for flat_idx in range(self.size):
            multi = flat_to_multi_index(flat_idx, self.shape)
            row = multi[mode]
            col_multi = [multi[i] for i in other_modes]
            if col_multi:
                col_strides = compute_strides(tuple(other_shape))
                col = multi_index_to_flat(tuple(col_multi), col_strides)
            else:
                col = 0
            result[(row, col)] = self.data[flat_idx]
        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if not isinstance(k, int):
            raise TypeError("k must be integer")
        if k < 0 or k >= self.ndim - 1:
            raise ValueError("k out of range")
        left = compute_size(self.shape[:k + 1])
        right = compute_size(self.shape[k + 1:])
        return DenseTensor((left, right), self.data.copy())

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
        sq_sum = 0.0
        for x in self.data:
            sq_sum = sq_sum + x * x
        return math.sqrt(sq_sum)

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            raise TypeError("other must have shape and data")
        check_shapes_match(self.shape, other.shape)
        data = [a + b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            raise TypeError("other must have shape and data")
        check_shapes_match(self.shape, other.shape)
        data = [a - b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError("scalar must be int or float")
        data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self * (-1)

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
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            return False
        if self.shape != other.shape:
            return False
        for a, b in zip(self.data, other.data):
            diff = abs(a - b)
            max_val = max(abs(a), abs(b))
            if diff > atol + rtol * max_val:
                return False
        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        def build(current_shape, offset):
            if len(current_shape) == 1:
                return self.data[offset:offset + current_shape[0]]
            result = []
            step = compute_size(current_shape[1:])
            for i in range(current_shape[0]):
                result.append(build(current_shape[1:], offset + i * step))
            return result

        return build(self.shape, 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.to_nested_list()})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()

