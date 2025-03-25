from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """
    # the general formula is that it is [s1 * ind1 + s2 * ind2, s3 * ind3
    return np.dot(index, strides)

def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = ordinal % shape[i]
        ordinal = ordinal // shape[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    翻译：将一个较大张量的索引 big_index（维度为 big_shape）转换为
    一个较小张量的索引 out_index（维度为 shape），遵循广播规则。
    在这种情况下，big_shape 可能比 shape 有更多的维度或更大的尺寸。
    额外的维度可能需要映射为0或被移除。
    
    功能：基于广播机制，将大张量的 index 映射到小张量的 index 上
    
    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    for i in range(len(shape)):
        offset = i + len(big_shape) - len(shape)
        out_index[i] = big_index[offset] if shape[i] != 1 else 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    功能：生成 broadcast 后的 dim
    
    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    
    Explanation:
        - 当两个数组的形状在某个维度上相等，或者其中一个维度为1时，可以进行广播
        - 广播规则：
            - 如果两个数组的形状在某个维度上相等，则该维度上的值保持不变
            - 如果其中一个数组的形状在某个维度上为1，则该维度上的值将被扩展为与另一个数组在该维度上的值相同
            - 如果两个数组的形状在某个维度上不相等且都不为1，则抛出IndexingError异常
        - 另一版的规则（来自 youtube）
            - 规则 1：大小为 1 的维度可以与任何维度进行广播。
            - 规则 2：可以使用view操作添加额外的大小为 1 的维度。
            - 规则 3：zip操作会自动添加起始的大小为 1 的维度
    
    Example:
        - A	            B	        结果
        (3, 4, 5)	    (3, 1, 5)	(3, 4, 5)
        (3, 4, 1)	    (3, 1, 5)	(3, 4, 5)
        (3, 4, 1)	    (1, 5)	    (3, 4, 5)
        (3, 4, 1)	    (3, 5)	    Fail
        
    Note from YouTuBe - 广播的作用
        - 作用如下：
            - Apply same operation multiple times
            - Avoid loops and writes
            - Save memory
        - 例如，对于 [1,2,3] + 10 这个运算，有两种原始实现方式
            - 原始实现 1：
                - 实现：转换成 [1,2,3] + [10,10,10]
                - 缺点：需要额外的内存空间
            - 原始实现 2：
                - 实现：for 循环给 [1,2,3] 每个位置上 + 10
                - 缺点：额外的运算，导致 graph 变得复杂
            - 广播可以解决这个问题，从而方便地实现下面的 zip

                a = tensor([1, 2, 4])
                b = tensor([3, 2])
                out = zeros((3, 2))
                for i in range(3):
                    for j in range(2):
                        out.data[i][j] = a[i] + b[j]

            
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    
    # 补齐维度，使两个形状长度相同
    while len(shape1) < len(shape2):
        shape1.insert(0, 1)
    while len(shape2) < len(shape1):
        shape2.insert(0, 1)
    
    result = []
    for s1, s2 in zip(shape1, shape2):
        if s1 == s2:
            result.append(s1)
        elif s1 == 1:
            result.append(s2)
        elif s2 == 1:
            result.append(s1)
        else:
            raise IndexingError(f"Shapes {shape1} and {shape2} cannot be broadcasted")
    return tuple(result)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"
        
        new_shape = tuple([self.shape[o] for o in order])
        new_strides = tuple([self.strides[o] for o in order])
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
