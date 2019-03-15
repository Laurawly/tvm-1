# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison, unused-argument
"""Non-maximum suppression operator"""
import math
import tvm

from tvm import api
from tvm.intrin import if_then_else
from topi.vision import non_max_suppression, get_valid_counts
from ..util import get_const_tuple


def get_valid_counts_pre(data, flag, idx, score_threshold):
    """Low level IR to get valid count of bounding boxes
    given a score threshold. Also moves valid boxes to the
    top of input data.

    Parameters
    ----------
    data: Buffer
        3D Buffer with shape [batch_size, num_anchors, 6], output of nms.

    flag : Buffer
        2D Buffer of flag indicating valid data with shape [batch_size, num_anchors].

    idx : Buffer
        2D Buffer of valid data indices with shape [batch_size, num_anchors].

    score_threshold: float32
        Lower limit of score for valid bounding boxes.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]

    ib = tvm.ir_builder.create()

    data = ib.buffer_ptr(data)
    flag = ib.buffer_ptr(flag)
    idx = ib.buffer_ptr(idx)
    score_threshold = tvm.make.node("FloatImm", dtype="float32", value=score_threshold)

    max_threads = int(math.sqrt(tvm.target.current_target(allow_none=False).max_num_threads))
    nthread_tx = max_threads
    nthread_bx = batch_size * num_anchors // max_threads + 1
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx

    with ib.if_scope(tid < batch_size * num_anchors):
        i = tid / num_anchors # number of batches
        j = tid % num_anchors # number of anchors
        base_idx = i * num_anchors * box_data_length
        with ib.if_scope(data[base_idx + j * box_data_length + 1] > score_threshold):
            flag[tid] = 1
            idx[tid] = 1
        with ib.else_scope():
            flag[tid] = 0
            idx[tid] = 0

    with ib.if_scope(tid < batch_size):
        with ib.for_range(0, num_anchors) as k:
            with ib.if_scope(k > 0):
                idx[tid * num_anchors + k] += idx[tid * num_anchors + k - 1]
            ib.emit(tvm.make.Call(None, 'tvm_storage_sync',
                                  tvm.convert(['shared']),
                                  tvm.expr.Call.Intrinsic, None, 0))

    return ib.get()


def get_valid_counts_ir(data, flag, idx, valid_count, out):
    """Low level IR to get valid count of bounding boxes
    given a score threshold. Also moves valid boxes to the
    top of input data.

    Parameters
    ----------
    data : Buffer
        Input data. 3-D Buffer with shape [batch_size, num_anchors, 6].

    flag : Buffer
        2D Buffer of flag indicating valid data with shape [batch_size, num_anchors].

    idx : Buffer
        2D Buffer of valid data indices with shape [batch_size, num_anchors].

    valid_count : Buffer
        1-D buffer for valid number of boxes.

    out : Buffer
        Rearranged data buffer.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.ir_builder.create()

    data = ib.buffer_ptr(data)
    flag = ib.buffer_ptr(flag)
    idx = ib.buffer_ptr(idx)
    valid_count = ib.buffer_ptr(valid_count)
    out = ib.buffer_ptr(out)

    max_threads = int(tvm.target.current_target(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = batch_size * num_anchors * elem_length // max_threads + 1
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx

    with ib.if_scope(tid < batch_size * num_anchors * elem_length):
        out[tid] = -1.0
    with ib.if_scope(tid < batch_size * num_anchors):
        i = tid / num_anchors # number of batches
        j = tid % num_anchors # number of anchors
        base_idx = i * num_anchors * elem_length
        with ib.if_scope(flag[tid] > 0):
            with ib.for_range(0, elem_length) as k:
                out[base_idx + (idx[tid] - 1) * elem_length + k] =\
                data[base_idx + j * elem_length + k]
        valid_count[i] = idx[i * num_anchors + num_anchors - 1]

    return ib.get()


@get_valid_counts.register(["cuda", "gpu"])
def get_valid_counts_gpu(data, score_threshold=0):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    Returns
    -------
    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.

    out_tensor : tvm.Tensor
        Rearranged data tensor.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    temp_flag_buf = api.decl_buffer(
        (batch_size, num_anchors,), "int32", "temp_flag", data_alignment=8)
    temp_idx_buf = api.decl_buffer(
        (batch_size, num_anchors,), "int32", "temp_idx", data_alignment=8)
    data_buf = api.decl_buffer(
        data.shape, data.dtype, "data_buf", data_alignment=8)
    temp_flag, temp_idx = \
        tvm.extern([(batch_size, num_anchors,), (batch_size, num_anchors,)], [data],
                   lambda ins, outs: get_valid_counts_pre(
                       ins[0], outs[0], outs[1], score_threshold),
                   dtype=["int32", "int32"],
                   out_buffers=[temp_flag_buf, temp_idx_buf],
                   name="get_valid_counts_phase_one")

    valid_count, out_tensor = \
	tvm.extern([(batch_size,), data.shape], [data, temp_flag, temp_idx],
            lambda ins, outs: get_valid_counts_ir(
                ins[0], ins[1], ins[2], outs[0], outs[1]),
            dtype=["int32", data.dtype],
            in_buffers=[data_buf, temp_flag_buf, temp_idx_buf],
            tag="get_valid_counts")

    return [valid_count, out_tensor]


def sort_ir(data, index, output):
    """Low level IR to do sorting on the GPU, same usage as tvm.contrib.sort.argsort on the CPU.

    Parameters
    ----------
    data: Buffer
        2D Buffer of input boxes' score with shape [batch_size, num_anchors].

    index : Buffer
        1D Buffer of number of valid number of boxes.

    output : Buffer
        2D Output buffer of indicies of sorted tensor with shape [batch_size, num_anchors].

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    assert data.dtype == "float32", "Currently only supports input dtype to be float32"
    batch, num_anchors = get_const_tuple(data.shape)
    max_threads = int(tvm.target.current_target(allow_none=False).max_num_threads)
    ib = tvm.ir_builder.create()
    p_data = ib.buffer_ptr(data)
    p_index = ib.buffer_ptr(index)
    p_out = ib.buffer_ptr(output)
    nthread_tx = max_threads
    nthread_bx = num_anchors // max_threads + 1
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("vthread")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "virtual_thread", nthread_bx)
    tid = bx * nthread_tx + tx
    temp_data = ib.allocate("float32", (1,), name="temp_data", scope="local")
    temp_index = ib.allocate("int32", (1,), name="temp_index", scope="local")

    with ib.for_range(0, batch, for_type="unroll") as b:
        start = b * num_anchors
        with ib.if_scope(tid < num_anchors):
            p_out[start + tid] = tid
        # OddEvenTransposeSort
        with ib.for_range(0, p_index[b]) as k:
            with ib.if_scope(tid < (p_index[b] + 1) // 2):
                offset = start + 2 * tid + (k % 2)
                with ib.if_scope( \
                        tvm.all(offset + 1 < p_index[0], p_data[offset] < p_data[offset + 1])):
                    temp_data[0] = p_data[offset]
                    p_data[offset] = p_data[offset + 1]
                    p_data[offset + 1] = temp_data[0]
                    temp_index[0] = p_out[offset]
                    p_out[offset] = p_out[offset + 1]
                    p_out[offset + 1] = temp_index[0]
            ib.emit(tvm.make.Call(None, 'tvm_storage_sync',
                                  tvm.convert(['shared']),
                                  tvm.expr.Call.Intrinsic, None, 0))

    return ib.get()

def nms_ir(data, sorted_index, valid_count, out, box_indices,
           max_output_size, iou_threshold, force_suppress, 
           top_k, coord_start, id_index):
    """Low level IR routing for transform location in multibox_detection operator.

    Parameters
    ----------
    data : Buffer
        Buffer of output boxes with class and score.

    sort_index : Buffer
        Buffer of output box indexes sorted by score.

    valid_count : Buffer
        Buffer of number of valid output boxes.

    out : Buffer
        Output buffer.

    max_output_size : int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : float
        Overlapping(IoU) threshold to suppress object with smaller score.

    force_suppress : boolean
        Whether to suppress all detections regardless of class_id.

    top_k : int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : int
        Start index of the consecutive 4 coordinates.

    id_index : int
        index of the class categories, -1 to disable.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
        """Calculate overlap of two boxes.
        """
        w = tvm.max(0.0, tvm.min(out_tensor[box_a_idx + 2], out_tensor[box_b_idx + 2])
                    - tvm.max(out_tensor[box_a_idx], out_tensor[box_b_idx]))
        h = tvm.max(0.0, tvm.min(out_tensor[box_a_idx + 3], out_tensor[box_b_idx + 3])
                    - tvm.max(out_tensor[box_a_idx + 1], out_tensor[box_b_idx + 1]))
        i = w * h
        u = (out_tensor[box_a_idx + 2] - out_tensor[box_a_idx]) * \
            (out_tensor[box_a_idx + 3] - out_tensor[box_a_idx + 1]) + \
            (out_tensor[box_b_idx + 2] - out_tensor[box_b_idx]) * \
            (out_tensor[box_b_idx + 3] - out_tensor[box_b_idx + 1]) - i
        return tvm.expr.Select(u <= 0.0, 0.0, i / u)

    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]

    ib = tvm.ir_builder.create()

    data = ib.buffer_ptr(data)
    sorted_index = ib.buffer_ptr(sorted_index)
    valid_count = ib.buffer_ptr(valid_count)
    out = ib.buffer_ptr(out)
    box_indices = ib.buffer_ptr(box_indices)
    num_valid_boxes = ib.allocate("int32", (1,), name="num_valid_boxes", scope="local")

    max_threads = int(math.sqrt(
        tvm.target.current_target(allow_none=False).max_num_threads))
    nthread_tx = max_threads
    nthread_bx = num_anchors // max_threads + 1
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    k = bx * max_threads + tx

    iou_threshold = tvm.make.node("FloatImm", dtype="float32", value=iou_threshold)
    top_k = tvm.make.node("IntImm", dtype="int32", value=top_k)
    coord_start = tvm.make.node("IntImm", dtype="int32", value=coord_start)
    id_index = tvm.make.node("IntImm", dtype="int32", value=id_index)
    force_suppress = tvm.make.node("IntImm", dtype="int32", value=1 if force_suppress else 0)

    with ib.for_range(0, batch_size, for_type="unroll") as i:
        base_idx = i * num_anchors * box_data_length
        with ib.if_scope(tvm.all(iou_threshold > 0, valid_count[i] > 0)):
            # Reorder output
            nkeep = if_then_else( \
                    tvm.all(top_k > 0, top_k < valid_count[i]),
                    top_k, valid_count[i])
            with ib.for_range(0, nkeep) as j:
                with ib.if_scope(k < box_data_length):
                    out[(base_idx + j * box_data_length + k)] = \
                    data[(base_idx + sorted_index[i * num_anchors + j] \
                    * box_data_length + k)]
                box_indices[i * num_anchors + j] = sorted_index[i * num_anchors + j]
            with ib.if_scope(tvm.all(top_k > 0, top_k < valid_count[i])):
                with ib.for_range(0, valid_count[i] - nkeep) as j:
                    with ib.if_scope(k < box_data_length):
                        out[(base_idx + (j + nkeep) * box_data_length + k)] = -1.0
                    box_indices[i * num_anchors + (j + nkeep)] = -1
            # Apply nms
            with ib.for_range(0, valid_count[i]) as j:
                offset_j = j * box_data_length
                with ib.if_scope(out[base_idx + offset_j] >= 0):
                    with ib.if_scope(k < valid_count[i]):
                        offset_k = k * box_data_length
                        with ib.if_scope(tvm.all(k > j, out[base_idx + offset_k] >= 0, \
						 tvm.any(force_suppress > 0, id_index < 0, \
                                                         out[base_idx + offset_j] == \
                                                         out[base_idx + offset_k]))):
                            iou = calculate_overlap(out, base_idx + offset_k + coord_start,
                                                    base_idx + offset_j + coord_start)
                            with ib.if_scope(iou >= iou_threshold):
                                out[base_idx + offset_k] = -1.0
                                box_indices[i * num_anchors + k] = -1
                ib.emit(tvm.make.Call(None, 'tvm_storage_sync',
                                      tvm.convert(['shared']),
                                      tvm.expr.Call.Intrinsic, None, 0))
        with ib.else_scope():
            with ib.for_range(0, valid_count[i]) as j:
                offset_j = j * box_data_length
                with ib.if_scope(k < box_data_length):
                    out[(base_idx + offset_j + k)] = data[base_idx + offset_j + k]
                box_indices[i * num_anchors + j] = j
        # Set invalid entry to be -1
        with ib.for_range(0, num_anchors - valid_count[i]) as j:
            with ib.if_scope(k < box_data_length):
                out[base_idx + (j + valid_count[i]) * box_data_length + k] = -1.0
            box_indices[i * num_anchors + j + valid_count[i]] = -1
        # Only return max_output_size number of valid boxes
        num_valid_boxes[0] = 0
        with ib.if_scope(max_output_size > 0):
            with ib.for_range(0, valid_count[i]) as j:
                offset_j = j * box_data_length
                with ib.if_scope(out[base_idx + offset_j] >= 0):
                    with ib.if_scope(num_valid_boxes[0] == max_output_size):
                        with ib.if_scope(k < box_data_length):
                            out[base_idx + offset_j + k] = -1.0
                        box_indices[i * num_anchors + j] = -1
                    with ib.else_scope():
                        num_valid_boxes[0] += 1
                ib.emit(tvm.make.Call(None, 'tvm_storage_sync',
                                      tvm.convert(['shared']),
                                      tvm.expr.Call.Intrinsic, None, 0))

    return ib.get()


def invalid_to_bottom_pre(data, flag, idx):
    """Low level IR to rearrange nms output to move all valid entries to top.

    Parameters
    ----------
    data: Buffer
        3D Buffer with shape [batch_size, num_anchors, 6], output of nms.

    flag : Buffer
        1D Buffer of flag indicating valid data with [num_anchors].

    idx : Buffer
        1D Buffer of valid data indices with [num_anchors].

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.ir_builder.create()

    data = ib.buffer_ptr(data)
    flag = ib.buffer_ptr(flag)
    idx = ib.buffer_ptr(idx)

    max_threads = int(math.sqrt(
        tvm.target.current_target(allow_none=False).max_num_threads))
    nthread_tx = max_threads
    nthread_bx = num_anchors // max_threads + 1
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    j = bx * max_threads + tx

    with ib.for_range(0, batch_size, for_type="unroll") as i:
        base_idx = i * num_anchors * elem_length
        with ib.if_scope(j < num_anchors):
            with ib.if_scope(data[base_idx + j * elem_length] >= 0):
                flag[i * num_anchors + j] = 1
                idx[i * num_anchors + j] = 1
            with ib.else_scope():
                flag[i * num_anchors + j] = 0
                idx[i * num_anchors + j] = 0

    with ib.if_scope(j < batch_size):
        with ib.for_range(0, num_anchors) as k:
            with ib.if_scope(k > 0):
                idx[j * num_anchors + k] += idx[j * num_anchors + k - 1]
    return ib.get()


def invalid_to_bottom_ir(data, flag, idx, out):
    """Low level IR to rearrange nms output to move all valid entries to top.

    Parameters
    ----------
    data: Buffer
        3D Buffer with shape [batch_size, num_anchors, 6], output of nms.

    flag : Buffer
        1D Buffer of flag indicating valid data with [num_anchors].

    idx : Buffer
        1D Buffer of valid data indices with [num_anchors].

    out : Buffer
        3D Buffer of rearranged nms output with shape [batch_size, num_anchors, 6].

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.ir_builder.create()

    data = ib.buffer_ptr(data)
    flag = ib.buffer_ptr(flag)
    idx = ib.buffer_ptr(idx)
    out = ib.buffer_ptr(out)

    max_threads = int(math.sqrt(
        tvm.target.current_target(allow_none=False).max_num_threads))
    nthread_tx = max_threads
    nthread_bx = num_anchors // max_threads + 1
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    j = bx * max_threads + tx

    with ib.for_range(0, batch_size, for_type="unroll") as i:
        base_idx = i * num_anchors * elem_length
        with ib.if_scope(j < num_anchors):
            with ib.for_range(0, elem_length) as k:
                out[base_idx + j * elem_length + k] = -1.0
            with ib.if_scope(flag[i * num_anchors + j] > 0):
                with ib.for_range(0, elem_length) as k:
                    out[base_idx + (idx[i * num_anchors + j] - 1) * elem_length + k] \
                    = data[base_idx + j * elem_length + k]
    return ib.get()


@non_max_suppression.register(["cuda", "gpu"])
def non_max_supression_gpu(data, valid_count, max_output_size=-1,
                           iou_threshold=0.5, force_suppress=False, top_k=-1,
                           coord_start=2, score_index=1, id_index=0,
                           return_indices=True, invalid_to_bottom=False):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.

    max_output_size : optional, int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : optional, float
        Non-maximum suppression threshold.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id.

    top_k : optional, int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : required, int
        Start index of the consecutive 4 coordinates.

    score_index : optional, int
        Index of the scores/confidence of boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    return_indices : boolean
        Whether to return box indices in input data.

    invalid_to_bottom : optional, boolean
        Whether to move all valid bounding boxes to the top.

    Returns
    -------
    out : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].

    Example
    --------
    .. code-block:: python

        # An example to use nms
        dshape = (1, 5, 6)
        data = tvm.placeholder(dshape, name="data")
        valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
        iou_threshold = 0.7
        force_suppress = True
        top_k = -1
        out = non_max_supression(data=data, valid_count=valid_count, iou_threshold=iout_threshold,
                                 force_suppress=force_supress, top_k=top_k, return_indices=False)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
        s = topi.generic.schedule_nms(out)
        f = tvm.build(s, [data, valid_count, out], "cuda")
        ctx = tvm.gpu(0)
        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f(tvm_data, tvm_valid_count, tvm_out)
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]

    valid_count_dtype = "int32"
    valid_count_buf = api.decl_buffer(valid_count.shape, valid_count_dtype,
                                      "valid_count_buf", data_alignment=4)
    score_axis = score_index
    score_shape = (batch_size, num_anchors)
    score_tensor = tvm.compute(score_shape, lambda i, j: data[i, j, score_axis])
    score_tensor_buf = api.decl_buffer(score_tensor.shape, data.dtype,
                                       "score_tensor_buf", data_alignment=8)

    sort_tensor_dtype = "int32"
    sort_tensor_buf = api.decl_buffer(score_shape, sort_tensor_dtype,
                                      "sort_tensor_buf", data_alignment=8)

    sort_tensor = \
        tvm.extern(score_shape,
                   [score_tensor, valid_count],
                   lambda ins, outs: sort_ir(
                       ins[0], ins[1], outs[0]),
                   dtype=sort_tensor_dtype,
                   in_buffers=[score_tensor_buf, valid_count_buf],
                   out_buffers=sort_tensor_buf,
                   name="nms_sort")

    data_buf = api.decl_buffer(
        data.shape, data.dtype, "data_buf", data_alignment=8)

    out_buf = api.decl_buffer(
        data.shape, data.dtype, "out_buf", data_alignment=8)

    box_indices_buf = api.decl_buffer(
        (batch_size, num_anchors), "int32", "box_indices_buf", data_alignment=8)

    out, box_indices = \
        tvm.extern([data.shape, (batch_size, num_anchors)],
                   [data, sort_tensor, valid_count],
                   lambda ins, outs: nms_ir(
                       ins[0], ins[1], ins[2], outs[0], outs[1],
                       max_output_size, iou_threshold, force_suppress,
                       top_k, coord_start, id_index),
                   dtype=[data.dtype, "int32"],
                   in_buffers=[data_buf, sort_tensor_buf, valid_count_buf],
                   tag="nms")

    if return_indices:
        return box_indices

    if invalid_to_bottom:
        output_buf = api.decl_buffer(
            data.shape, data.dtype, "output_buf", data_alignment=8)
        temp_flag_buf = api.decl_buffer(
            (batch_size, num_anchors,), valid_count_dtype, "temp_flag", data_alignment=8)
        temp_idx_buf = api.decl_buffer(
            (batch_size, num_anchors,), valid_count_dtype, "temp_idx", data_alignment=8)
        temp_flag, temp_idx = tvm.extern([(batch_size, num_anchors,), \
                                          (batch_size, num_anchors,)], [out],
                                         lambda ins, outs: invalid_to_bottom_pre(
                                             ins[0], outs[0], outs[1]),
                                         dtype=["int32", "int32"],
                                         in_buffers=[out_buf],
                                         out_buffers=[temp_flag_buf, temp_idx_buf],
                                         name="invalid_to_bottom_phase_one")

        output = tvm.extern([data.shape], [out, temp_flag, temp_idx],
                            lambda ins, outs: invalid_to_bottom_ir(
                                ins[0], ins[1], ins[2], outs[0]),
                            dtype=[data.dtype],
                            in_buffers=[out_buf, temp_flag_buf, temp_idx_buf],
                            out_buffers=[output_buf],
                            tag="invalid_to_bottom")
        return output

    return out
