"""Patches for coremltools to support Qwen3.5 DeltaNet ops.

Only registers ops that are genuinely missing from coremltools 8.3.
"""

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
import coremltools.converters.mil.frontend.torch.ops as ct_ops


# --- Fix: _cast crashes on non-scalar numpy arrays ---

def _cast_fixed(context, node, dtype, dtype_name):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    if x.can_be_folded_to_const():
        val = x.val
        if hasattr(val, "item"):
            try:
                val = val.item()
            except (ValueError, AttributeError):
                val = dtype(np.array(val).flat[0])
        if not isinstance(val, dtype):
            res = mb.const(val=dtype(val), name=node.name)
        else:
            res = x
    elif len(x.shape) > 0:
        x = mb.squeeze(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    else:
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)


ct_ops._cast = _cast_fixed


# --- Missing ops ---

@register_torch_op
def diff(context, node):
    """torch.diff(x, n=1, dim=-1) -> x[..., 1:] - x[..., :-1]"""
    inputs = _get_inputs(context, node)
    x = inputs[0]
    dim = inputs[2].val if len(inputs) > 2 and inputs[2] is not None else -1

    begin_after = [0] * len(x.shape)
    begin_after[dim] = 1
    end = list(x.shape)
    x_after = mb.slice_by_index(
        x=x, begin=begin_after, end=end, name=node.name + "_after"
    )

    begin = [0] * len(x.shape)
    end_before = list(x.shape)
    end_before[dim] = -1
    x_before = mb.slice_by_index(
        x=x, begin=begin, end=end_before, name=node.name + "_before"
    )

    res = mb.sub(x=x_after, y=x_before, name=node.name)
    context.add(res, node.name)


@register_torch_op
def new_ones(context, node):
    """Tensor.new_ones(shape) -> ones tensor."""
    inputs = _get_inputs(context, node)
    # inputs[0] = source tensor, inputs[1:] could be individual dims or a list
    raw_shape = []
    for i in inputs[1:]:
        val = i.val if hasattr(i, "val") else i
        if val is None:
            continue
        if isinstance(val, (list, tuple, np.ndarray)):
            raw_shape.extend([int(v) for v in val])
        elif isinstance(val, (int, float, np.integer)):
            raw_shape.append(int(val))
    # Infer dtype from source tensor
    src = inputs[0]
    res = mb.fill(shape=np.array(raw_shape, dtype=np.int32), value=1.0, name=node.name)
    context.add(res, node.name)


@register_torch_op
def chunk(context, node):
    """torch.chunk(x, chunks, dim) -> split into N equal parts."""
    inputs = _get_inputs(context, node)
    x = inputs[0]
    num_chunks = inputs[1].val
    dim = inputs[2].val if len(inputs) > 2 else 0

    total = x.shape[dim]
    split_size = total // num_chunks
    split_sizes = [split_size] * num_chunks
    remainder = total - split_size * num_chunks
    if remainder > 0:
        split_sizes[-1] += remainder

    res = mb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
    context.add(res, node.name)


# --- Fix: pad op crashes when pad arg is a list of individual Vars ---
# The internal _translate_torch_args does `pad.val` but pad is a list.
# Patch the internal helper that processes pad args.

import coremltools.converters.mil.frontend.torch.ops as _ops_module

# Find and patch the nested _translate_torch_args inside pad
_original_pad_func = _ops_module.pad.__wrapped__ if hasattr(_ops_module.pad, '__wrapped__') else None

@register_torch_op(override=True, torch_alias=["constant_pad_nd"])
def pad(context, node):
    """Patched pad that handles list pad arguments."""
    inputs = _get_inputs(context, node, expected=(2, 3, 4))
    x = inputs[0]
    pad_arg = inputs[1]

    # Handle pad being a list of individual MIL vars
    if isinstance(pad_arg, list):
        def _to_int(v):
            val = v.val if hasattr(v, "val") else v
            if hasattr(val, "item"):
                return int(val.item())
            return int(val)
        pad_vals = [_to_int(v) for v in pad_arg]
    elif hasattr(pad_arg, "val") and pad_arg.val is not None:
        pad_vals = pad_arg.val.tolist() if hasattr(pad_arg.val, "tolist") else list(pad_arg.val)
    else:
        pad_vals = None

    mode = "constant"
    value = 0.0
    if len(inputs) > 2:
        m = inputs[2]
        mode = m.val if hasattr(m, "val") and m.val is not None else m
    if len(inputs) > 3 and inputs[3] is not None:
        v = inputs[3]
        if hasattr(v, "val") and v.val is not None:
            value = float(v.val)
        elif v is not None:
            try:
                value = float(v)
            except (TypeError, ValueError):
                value = 0.0

    if pad_vals is not None:
        # Torch pad: [left, right, top, bottom, ...] (innermost first)
        # CoreML pad: [top, bottom, left, right, ...] (outermost first) per-axis [before, after]
        n_pad = len(pad_vals)
        rank = len(x.shape)

        # Check for negative pads — torch F.pad handles these as slices
        has_negative = any(p < 0 for p in pad_vals)
        if has_negative:
            # Convert negative pads to slice_by_index + positive pad
            begin = [0] * rank
            end = list(x.shape)
            pos_pads = [0] * (2 * rank)
            for i in range(0, n_pad, 2):
                dim_from_end = i // 2
                dim = rank - 1 - dim_from_end
                left, right = pad_vals[i], pad_vals[i + 1]
                if left < 0:
                    begin[dim] = -left  # slice off from left
                    left = 0
                if right < 0:
                    end[dim] = end[dim] + right  # slice off from right
                    right = 0
                pos_pads[2 * dim] = left
                pos_pads[2 * dim + 1] = right

            x = mb.slice_by_index(x=x, begin=begin, end=end, name=node.name + "_negpad_slice")
            if any(p > 0 for p in pos_pads):
                res = mb.pad(x=x, pad=np.array(pos_pads, dtype=np.int32), mode=mode,
                            constant_val=value, name=node.name)
            else:
                res = x
        else:
            # Build CoreML pad array: [before_dim0, after_dim0, before_dim1, after_dim1, ...]
            coreml_pad = [0] * (2 * rank)
            for i in range(0, n_pad, 2):
                dim_from_end = i // 2
                dim = rank - 1 - dim_from_end
                coreml_pad[2 * dim] = pad_vals[i]      # before
                coreml_pad[2 * dim + 1] = pad_vals[i + 1]  # after

            res = mb.pad(
                x=x,
                pad=np.array(coreml_pad, dtype=np.int32),
                mode=mode,
                constant_val=value,
                name=node.name,
            )
    else:
        res = mb.pad(x=x, pad=pad_arg, mode=mode, constant_val=value, name=node.name)

    context.add(res, node.name)


# --- Fix: reshape gets list of Vars instead of shape tuple ---

@register_torch_op(override=True)
def reshape(context, node):
    """Fix reshape when shape is a list of individual MIL Vars."""
    inputs = _get_inputs(context, node)
    x = inputs[0]
    shape_arg = inputs[1]

    if isinstance(shape_arg, list):
        # List of individual dimension Vars — extract values
        shape_vals = []
        all_const = True
        for v in shape_arg:
            if hasattr(v, "val") and v.val is not None:
                val = v.val
                shape_vals.append(int(val.item()) if hasattr(val, "item") else int(val))
            else:
                all_const = False
                shape_vals.append(v)

        if all_const:
            res = mb.reshape(x=x, shape=np.array(shape_vals, dtype=np.int32), name=node.name)
        else:
            # Dynamic dims — concat into a shape tensor
            const_parts = []
            for v in shape_vals:
                if isinstance(v, int):
                    const_parts.append(mb.const(val=np.array([v], dtype=np.int32)))
                else:
                    const_parts.append(mb.expand_dims(x=v, axes=[0], name=node.name + "_edim"))
            shape_tensor = mb.concat(values=const_parts, axis=0, name=node.name + "_shape")
            res = mb.reshape(x=x, shape=shape_tensor, name=node.name)
    else:
        # Standard case — shape is a single var or const
        if hasattr(shape_arg, "val") and shape_arg.val is not None:
            shape = shape_arg.val
            if hasattr(shape, "tolist"):
                shape = shape.tolist()
            res = mb.reshape(x=x, shape=np.array(shape, dtype=np.int32), name=node.name)
        else:
            res = mb.reshape(x=x, shape=shape_arg, name=node.name)

    context.add(res, node.name)


# --- Fix: view also gets list of Vars (same issue as reshape) ---

@register_torch_op(torch_alias=["view_copy", "_unsafe_view"], override=True)
def view(context, node):
    """Fix view when shape is a list of individual MIL Vars (same fix as reshape)."""
    reshape(context, node)


# --- Fix: slice op gets list of Vars with inhomogeneous shapes ---

@register_torch_op(override=True)
def slice(context, node):
    """Fix slice when arguments are MIL Vars instead of constants."""
    inputs = _get_inputs(context, node)
    x = inputs[0]
    dim = inputs[1].val if hasattr(inputs[1], "val") else inputs[1]
    start = inputs[2].val if hasattr(inputs[2], "val") and inputs[2].val is not None else 0
    end = inputs[3].val if len(inputs) > 3 and hasattr(inputs[3], "val") and inputs[3].val is not None else x.shape[dim]
    step = inputs[4].val if len(inputs) > 4 and hasattr(inputs[4], "val") and inputs[4].val is not None else 1

    # Convert numpy arrays to Python scalars
    if hasattr(dim, "item"):
        dim = dim.item()
    if hasattr(start, "item"):
        start = start.item()
    if hasattr(end, "item"):
        end = end.item()
    if hasattr(step, "item"):
        step = step.item()

    dim = int(dim)
    start = int(start)
    end = int(end) if end is not None else x.shape[dim]
    step = int(step)

    begin = [0] * len(x.shape)
    end_vals = list(x.shape)
    strides = [1] * len(x.shape)

    begin[dim] = start
    end_vals[dim] = end
    strides[dim] = step

    res = mb.slice_by_index(
        x=x, begin=begin, end=end_vals, stride=strides, name=node.name,
    )
    context.add(res, node.name)


print("coremltools patches applied (3 new ops + 4 bugfixes)")
