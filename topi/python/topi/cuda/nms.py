# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison, unused-argument
"""Non-maximum suppression operator"""
import math
import tvm

from tvm import api
from topi.vision import non_max_suppression
from ..util import get_const_tuple

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
    data: Buffer
        Buffer of output boxes with class and score.

    sort_result : Buffer
        Buffer of output box indexes sorted by score.

    valid_count : Buffer
        Buffer of number of valid output boxes.

    out : Buffer
        Output buffer.

    nms_threshold : float
        Non-maximum suppression threshold.

    force_suppress : boolean
        Whether to suppress all detections regardless of class_id.

    nms_topk : int
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

    max_threads = int(math.sqrt(
        tvm.target.current_target(allow_none=False).max_num_threads))
    ib = tvm.ir_builder.create()
    p_data = ib.buffer_ptr(data)
    p_sort_result = ib.buffer_ptr(sort_result)
    p_valid_count = ib.buffer_ptr(valid_count)
    p_out = ib.buffer_ptr(out)
    batch_size = out.shape[0]
    num_anchors = out.shape[1]
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
            nkeep = tvm.if_then_else( \
                    tvm.all(nms_topk_node > 0, nms_topk < p_valid_count[b]),
                    nms_topk, p_valid_count[b])
            with ib.for_range(0, nkeep) as l:
                with ib.if_scope(i < 6):
                    p_out[(base_idx + l * 6 + i)] = \
                            p_data[(base_idx + p_sort_result[b * num_anchors + l] * 6 + i)]
            with ib.if_scope(tvm.all(nms_topk_node > 0, nms_topk < p_valid_count[b])):
                with ib.for_range(0, p_valid_count[b] - nkeep) as l:
                    with ib.if_scope(i < 6):
                        p_out[(base_idx + (l + nkeep) * 6 + i)] = -1.0
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
            with ib.for_range(0, p_valid_count[b]) as c:
                with ib.if_scope(i < 6):
                    p_out[(base_idx + c * 6 + i)] = p_data[base_idx + c * 6 + i]
        # Set invalid entry to be -1
        with ib.for_range(0, num_anchors - p_valid_count[b]) as c:
            with ib.if_scope(i < 6):
                p_out[base_idx + (c + p_valid_count[b]) * 6 + i] = -1.0
    body = ib.get()
    return body


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

    return_indices : boolean
        Whether to return box indices in input data.

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
        out = nms(data, valid_count, iou_threshold, force_suppress, topk)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
        s = topi.generic.schedule_nms(out)
        f = tvm.build(s, [data, valid_count, out], "llvm")
        ctx = tvm.cpu()
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
    score_tensor = tvm.compute(
        score_shape, lambda i, j: data[i, j, 1], name="score_tensor")
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

    out = \
        tvm.extern(data.shape,
                   [data, sort_tensor, valid_count],
                   lambda ins, outs: nms_ir(
                       ins[0], ins[1], ins[2], outs[0], outs[1],
                       max_output_size, iou_threshold, force_suppress,
                       top_k, coord_start, id_index),
                   dtype=[data.dtype, "int32"],
                   in_buffers=[data_buf, sort_tensor_buf, valid_count_buf],
                   tag="nms")
    return out
