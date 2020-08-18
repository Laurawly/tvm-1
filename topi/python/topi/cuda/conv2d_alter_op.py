# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Conv2D alter op and legalize functions for cuda backend"""

import logging
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm

from .. import nn
from ..util import get_const_tuple
from .conv2d_winograd import _infer_tile_size
from ..nn import conv2d_legalize

logger = logging.getLogger('topi')

@nn.conv2d_alter_layout.register(["cuda", "gpu"])
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current

    _, outs = relay.backend.compile_engine.select_implementation(
        relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target)
    workload = autotvm.task.get_workload(outs)
    if workload is None:
        # The best implementation is not an AutoTVM template,
        # we then assume it's not necessary to alter this op.
        return None
    cfg = dispatch_ctx.query(target, workload)
    # TODO(@were): This is not good hack, '.nvptx'
    if cfg.is_fallback and not workload[0].endswith('.nvptx'):  # if is fallback, clear query cache and return None
        autotvm.task.clear_fallback_cache(target, workload)
        return None

    topi_tmpl = workload[0]
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int('groups')
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data, kernel = tinfos
    out_dtype = out_type.dtype

    if topi_tmpl == 'conv2d_NCHW16c_OHWI16o.nvptx':
        new_attrs['data_layout'] = 'NCHW16c'
        N, CI, H, W = get_const_tuple(data.shape)
        new_attrs['kernel_layout'] = 'OIHW16i16o'
        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_NCHWc_int8.cuda":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        new_layout = 'NCHW4c'
        new_attrs["channels"] = CO
        new_attrs["data_layout"] = new_layout
        new_attrs['out_layout'] = new_layout
        new_attrs['kernel_layout'] = 'OIHW4o4i'
        ic_block_factor = oc_block_factor = 4

        # Store the same config for the altered operator (workload)
        new_data = te.placeholder((N, CI // ic_block_factor, H, W, ic_block_factor),
                                  dtype=data.dtype)
        new_kernel = te.placeholder((CO // oc_block_factor, CI // ic_block_factor, KH, KW, \
                                     oc_block_factor, ic_block_factor), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_layout, out_dtype],
            "conv2d_NCHWc_int8.cuda")
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_nchw_winograd.cuda":
        if dilation != (1, 1):
            logger.warning("Does not support weight pre-transform for dilated convolution.")
            return None

        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        # pre-compute weight transformation in winograd
        tile_size = _infer_tile_size(tinfos[0], tinfos[1])

        weight = relay.nn.contrib_conv2d_winograd_weight_transform(inputs[1],
                                                                   tile_size=tile_size)
        weight = relay.transpose(weight, axes=[0, 1, 3, 2])
        new_attrs['tile_size'] = tile_size
        new_attrs['channels'] = CO

        # Store the same config for the altered operator (workload)
        new_data = data
        new_weight = te.placeholder((KH + tile_size - 1, KW + tile_size - 1, CI, CO),
                                    dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_weight, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd_without_weight_transform.cuda")
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight, **new_attrs)

    if topi_tmpl in ('conv2d_nhwc_winograd_direct.cuda', 'conv2d_nhwc_winograd_tensorcore.cuda'):
        if dilation != (1, 1):
            logger.warning("Does not support weight pre-transform for dilated convolution.")
            return None

        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        N, H, W, CI = get_const_tuple(data.shape)
        KH, KW, _, CO = get_const_tuple(kernel.shape)

        # Pre-compute weight transformation in winograd
        if H % 8 == 0:
            tile_size = 4
        else:
            tile_size = 2
        kernel_transform = relay.transpose(inputs[1], axes=[3, 2, 0, 1])
        weight = relay.nn.contrib_conv2d_winograd_weight_transform(kernel_transform,
                                                                   tile_size=tile_size)
        weight = relay.transpose(weight, axes=[0, 1, 3, 2])
        new_attrs['tile_size'] = tile_size
        new_attrs['channels'] = CO
        # Store the same config for the altered operator (workload)
        new_data = data
        new_weight = te.placeholder((KH + tile_size - 1, KW + tile_size - 1, CI, CO),
                                    dtype=kernel.dtype)
        if topi_tmpl == "conv2d_nhwc_winograd_direct.cuda":
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_weight, strides, padding, dilation, out_dtype],
                "conv2d_nhwc_winograd_direct_without_weight_transform.cuda")
        elif topi_tmpl == "conv2d_nhwc_winograd_tensorcore.cuda":
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_weight, strides, padding, dilation, out_dtype],
                "conv2d_nhwc_winograd_tensorcore_without_weight_transform.cuda")
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight, **new_attrs)

    if topi_tmpl == "group_conv2d_NCHWc_int8.cuda":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        new_layout = 'NCHW4c'
        new_attrs["channels"] = CO
        new_attrs["data_layout"] = new_layout
        new_attrs['out_layout'] = new_layout
        new_attrs['kernel_layout'] = 'OIHW4o4i'
        ic_block_factor = oc_block_factor = 4

        # Store the same config for the altered operator (workload)
        new_data = te.placeholder((N, CI // ic_block_factor, H, W, ic_block_factor),
                                  dtype=data.dtype)
        new_kernel = te.placeholder((CO // oc_block_factor, CI // ic_block_factor // groups,
                                     KH, KW, oc_block_factor, ic_block_factor),
                                    dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, groups, out_dtype],
            "group_conv2d_NCHWc_int8.cuda")
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)

    return None

@conv2d_legalize.register("cuda")
def _conv2d_legalize(attrs, inputs, arg_types):
    """Legalizes Conv2D op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """

    # Dilation not supported yet. Return None if dilation is not (1, 1)
    dilation = attrs.get_int_tuple("dilation")
    if not (dilation[0] == 1 and dilation[1] == 1):
        return None

    # No legalization for depthwise convolutions yet.
    groups = attrs.get_int("groups")
    if groups != 1:
        return None

    # Collect the input tensors.
    data_tensor, kernel_tensor = arg_types[0], arg_types[1]
    data_dtype = data_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, kernel = inputs

    # Get the conv attrs
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    # Get data layout. Return None if not NCHW
    data_layout = attrs['data_layout']
    kernel_layout = attrs['kernel_layout']


    stride_w, stride_h = attrs.get_int_tuple('strides')

    if data_dtype in ['float16', 'float32'] and kernel_layout == 'OIHW':
        if data_dtype != 'float16':
            data = relay.cast(data, 'float16')
        kernel = relay.cast(kernel, 'float16')
        new_attrs['out_dtype'] = 'float32'
        kh, kw = attrs.get_int_tuple('kernel_size')
        batch, ic, height, width = get_const_tuple(data_tensor.shape)

        #if ic < 24:
        #    return None

        padding = attrs.get_int_tuple('padding')
        pad_h = pad_w = 0
        if sum(padding):
            if len(padding) == 4:
                pad_h = padding[0] + padding[2]
                pad_w = padding[1] + padding[3]
                data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (padding[0], padding[2]), (padding[1], padding[3])))
            elif len(padding) == 2:
                pad_h = padding[0] * 2
                pad_w = padding[1] * 2
                data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
            elif len(padding) == 1:
                pad_h = pad_w = padding[0] * 2
                data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[0], padding[0])))
            else:
                assert False
            new_attrs['padding'] = [0, 0, 0, 0]

        height += pad_h
        width += pad_w

        f_conv_dim = lambda indim, kdim, stride: (indim - kdim) // stride + 1

        oh = f_conv_dim(height, kh, stride_h)
        ow = f_conv_dim(width, kw, stride_w)
        oc = attrs.get_int('channels').value
        ow_changed = False

        #if ow % 32:
        #    diff0 = stride_w - (width - kw + 1) % stride_w
        #    diff1 = (32 - (ow + 1) % 32) * stride_w
        #    assert (width + diff0 + diff1 - kw + 1) // stride_w % 32 == 0
        #    assert (width + diff0 + diff1 - kw + 1) % stride_w == 0
        #    data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, diff0 + diff1)))
        #    ow_changed = True
        if not ((oh * ow % 32 == 0 and 32 % ow == 0) or ow % 32 == 0):
            first_h = stride_h - (height - kh) % stride_h
            first_w = stride_w - (width - kw) % stride_w
            max_diff_h = 32 - oh % 32
            max_diff_w = 32 - ow % 32
            diffh = diffw = 1e9
            for i in range(max_diff_h + 1):
                for j in range(max_diff_w + 1):
                    if (((oh + i) * (ow + j) % 32 == 0 and 32 % (ow + j) == 0) or ((ow + j) % 32 == 0)) and i + j < diffh + diffw:
                        def to_pad(padding, first, stride):
                            if padding == 0:
                                return 0
                            assert padding >= 1
                            return (padding - 1) * stride + first
                        diffh, diffw = to_pad(i, first_h, stride_h), to_pad(j, first_w, stride_w)
            #assert (height + diffh - kh + 1) * (width + diffw - kw + 1) % 32 == 0
            data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, diffh), (0, diffw)))
            ow_changed = True
        ic_split = 64
        ratio = 1e9
        ic_split = -1
        for to_split in [16, 32, 64]:
            if (to_split - ic % to_split) / to_split < ratio:
                ratio = (to_split - ic % to_split) / to_split
                ic_split = to_split
        if ic % ic_split:
            diff = ic_split - ic % ic_split
            data = relay.nn.pad(data, pad_width=((0, 0), (0, diff), (0, 0), (0, 0)))
            kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, diff), (0, 0), (0, 0)))
        oc_changed = False
        if oc % 32:
            diff = 32 - oc % 32
            kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
            new_attrs['channels'] = oc + diff
            oc_changed = True
        out = relay.nn.conv2d(data, kernel, **new_attrs)
        if ow_changed or oc_changed:
            begins = relay.const(tvm.nd.array([0, 0, 0, 0]))
            ends = relay.const(tvm.nd.array([batch, oc, oh, ow]))
            out = relay.strided_slice(out,
                                      begin=relay.const(tvm.nd.array([0, 0, 0, 0])),
                                      end=relay.const(tvm.nd.array([batch, oc, oh, ow])))
        return out

    # Pad input and output channels to use int8 schedule.
    if data_dtype in ['int8', 'uint8']:
        if data_layout == 'NCHW' and kernel_layout == "OIHW":
            oc_modified = False
            in_channel = data_tensor.shape[1].value
            out_channel = kernel_tensor.shape[0].value

            # Pad input channel
            if in_channel % 4 != 0:
                new_in_channel = ((in_channel + 4) // 4) * 4
                diff = new_in_channel - in_channel
                pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
                data = relay.nn.pad(data, pad_width=pad_width)
                kernel = relay.nn.pad(kernel, pad_width=pad_width)

            # Pad output channel
            new_out_channel = out_channel
            if out_channel % 4 != 0:
                new_out_channel = ((out_channel + 4) // 4) * 4
                diff = new_out_channel - out_channel
                kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
                oc_modified = True

            if oc_modified:
                new_attrs['channels'] = new_out_channel
                out = tvm.relay.nn.conv2d(data, kernel, **new_attrs)
                original_out_shape = [x.value for x in output_tensor.shape]
                out = relay.strided_slice(out, begin=relay.const([0, 0, 0, 0]),
                                          end=relay.const(original_out_shape))
            else:
                out = relay.nn.conv2d(data, kernel, **new_attrs)
            return out
    return None
