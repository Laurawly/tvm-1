"""Basic tensor operations."""
# pylint: disable=redefined-builtin
from __future__ import absolute_import as _abs
from . import _make
from ..expr import Tuple
from ... import nd as _nd
from ... import TVMContext as _TVMContext

# We create a wrapper function for each operator in the
# python side to call into the positional _make.OpName function.
#
# We make this decision so that we can:
# - Have declare python docstring for each function
# - Enable keyword arguments easily
# - Not put too much burden on FFI to support complicated features
#   like default value and keyword arguments

def log(data):
    """Compute elementwise log of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.log(data)


def exp(data):
    """Compute elementwise exp of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.exp(data)


def sqrt(data):
    """Compute elementwise sqrt of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sqrt(data)


def sigmoid(data):
    """Compute elementwise sigmoid of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sigmoid(data)


def floor(data):
    """Compute element-wise floor of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.floor(data)


def ceil(data):
    """Compute element-wise ceil of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.ceil(data)


def trunc(data):
    """Compute element-wise trunc of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.trunc(data)


def round(data):
    """Compute element-wise round of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.round(data)


def abs(data):
    """Compute element-wise absolute of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.abs(data)

def sign(data):
    """Compute element-wise absolute of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sign(data)

def tanh(data):
    """Compute element-wise tanh of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.tanh(data)


def negative(data):
    """Compute element-wise negative of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.negative(data)


def logical_not(data):
    """Compute element-wise logical not of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.logical_not(data)


def add(lhs, rhs):
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.

    Examples
    --------
    .. code:: python

      x = relay.Var("a") # shape is [2, 3]
      y = relay.Var("b") # shape is [2, 1]
      z = relay.add(x, y)  # result shape is [2, 3]
    """
    return _make.add(lhs, rhs)


def subtract(lhs, rhs):
    """Subtraction with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.subtract(lhs, rhs)


def multiply(lhs, rhs):
    """Multiplication with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.multiply(lhs, rhs)


def divide(lhs, rhs):
    """Division with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.divide(lhs, rhs)


def power(lhs, rhs):
    """Power with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.power(lhs, rhs)


def mod(lhs, rhs):
    """Mod with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.mod(lhs, rhs)


def logical_and(lhs, rhs):
    """logical AND with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.logical_and(lhs, rhs)


def logical_or(lhs, rhs):
    """logical OR with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.logical_or(lhs, rhs)


def equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs == rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.equal(lhs, rhs)


def not_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs != rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.not_equal(lhs, rhs)


def less(lhs, rhs):
    """Broadcasted elementwise test for (lhs < rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.less(lhs, rhs)


def less_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs <= rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.less_equal(lhs, rhs)


def greater(lhs, rhs):
    """Broadcasted elementwise test for (lhs > rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.greater(lhs, rhs)


def greater_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs >= rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.greater_equal(lhs, rhs)


def maximum(lhs, rhs):
    """Maximum with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.maximum(lhs, rhs)


def minimum(lhs, rhs):
    """Minimum with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.minimum(lhs, rhs)


def right_shift(lhs, rhs):
    """Right shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.right_shift(lhs, rhs)


def left_shift(lhs, rhs):
    """Left shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.left_shift(lhs, rhs)


def zeros(shape, dtype):
    """Fill array with zeros.

    Parameters
    ----------
    shape : tuple of int
        The shape of the target.

    dtype : data type
        The data type of the target.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    return _make.zeros(shape, dtype)


def zeros_like(data):
    """Returns an array of zeros, with same type and shape as the input.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.zeros_like(data)


def ones(shape, dtype):
    """Fill array with ones.

    Parameters
    ----------
    shape : tuple of int
        The shape of the target.

    dtype : data type
        The data type of the target.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    return _make.ones(shape, dtype)


def ones_like(data):
    """Returns an array of ones, with same type and shape as the input.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.ones_like(data)


def clip(a, a_min, a_max):
    """Clip the elements in `a` between `a_min` and `a_max`.
    `a_min` and `a_max` are cast to `a`'s dtype.

    Parameters
    ----------
    a : relay.Expr
        The input tensor.
    a_min : float
        The clip minimum.
    a_max : float
        The clip maximum.

    Returns
    -------
    result : relay.Expr
        `a` with elements clipped between `a_min` and `a_max`.

    Examples
    --------
    .. code:: python
      x = relay.Constant(tvm.nd.array([0, 1, 5, 3, 4, 2]))
      relay.clip(x, 1., 4.)
      # [1, 1, 4, 3, 4, 2]
    """
    return _make.clip(a, a_min, a_max)


def concatenate(data, axis):
    """Concatenate the input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        A list of tensors.
    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated tensor.
    """
    data = list(data)
    if not data:
        raise ValueError("relay.concatenate requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    return _make.concatenate(Tuple(data), axis)


def stack(data, axis):
    """Join a sequence of arrays along a new axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple(relay.Expr))
        A list of tensors.

    axis : int
        The axis in the result array along which the input arrays are stacked.

    Returns
    -------
    ret : relay.Expr
        The stacked tensor.
    """
    data = list(data)
    if not data:
        raise ValueError("relay.stack requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    return _make.stack(Tuple(data), axis)


def copy(data):
    """Copy a tensor.

    Parameters
    ----------
    data : relay.Expr
        The tensor to be copied.

    Returns
    -------
    result: relay.Expr
        The copied result.
    """
    return _make.copy(data)


def device_copy(data, src_dev, dst_dev):
    """Copy data from the source device to the destination device. This
    operator helps data transferring between difference contexts for
    heterogeneous execution.

    Parameters
    ----------
    data : tvm.relay.Expr
        The tensor to be copied.

    src_dev : Union[:py:class:`TVMContext`, str]
        The source device where the data is copied from.

    dst_dev : Union[:py:class:`TVMContext`, str]
        The destination device where the data is copied to.

    Returns
    -------
    result : tvm.relay.Expr
        The copied result.
    """
    if isinstance(src_dev, _TVMContext):
        src_dev = src_dev.device_type
    elif isinstance(src_dev, str):
        src_dev = _nd.context(src_dev).device_type
    else:
        raise ValueError("src_dev is expected to be the type of TVMContext or "
                         "str, but received %s" % (type(src_dev)))

    if isinstance(dst_dev, _TVMContext):
        dst_dev = dst_dev.device_type
    elif isinstance(dst_dev, str):
        dst_dev = _nd.context(dst_dev).device_type
    else:
        raise ValueError("dst_dev is expected to be the type of TVMContext or "
                         "str, but received %s" % (type(dst_dev)))
    return _make.device_copy(data, src_dev, dst_dev)


def shape_of(data, dtype="int32"):
    """Get shape of a tensor.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor.

    dtype : str, optional
        The target data type.

    Returns
    -------
    result : tvm.relay.Expr
        The shape tensor.
    """
    return _make.shape_of(data, dtype)
