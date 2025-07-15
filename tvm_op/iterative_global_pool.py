import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay, topi
import tvm.te


from tvm.ir import Attrs
import tvm.relay.op.op as _op
import copy
from tvm.target import generic_func, override_native_generic_func
import math
from tvm.ir import register_intrin_lowering, Op, register_op_attr

# Define the new operator in Relay
op_name = "iterative_global_pool"
relay.op.op.register(op_name)


# call default relation functions
def _rel(args, attrs):
    # func = attrs["relay_func"]
    # return relay.TupleType([relay.TensorType(attrs["output_shape"], args[1].dtype), relay.TensorType(func.body.checked_type.shape,func.body.checked_type.dtype)])
    # print("SSM Type REL:", args[1])
    # return args[1]
    return relay.TensorType(args[1].shape[:-1] + [1], args[0].dtype)
_op.get(op_name).add_type_rel("IterGlobalPoolTypeRel", _rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.INJECTIVE) # avoid tvm's error if ssm followed by pooling
_op.register_stateful(op_name, True)

# Avoid name conflictsin C Code
_identifier_idx = 1

def ceildiv(a, b):
    return -(a // -b)

#bf16: use bfloat 16 for state storage
def iterative_global_pool(x, pool_type: str, pool_size: int, latent_dim: list[int], stride: int, bf16=False):
    global _identifier_idx
    num_of_latent_state = ceildiv(pool_size, stride)
    latent_state_shape = tuple([*latent_dim, num_of_latent_state])
    if bf16:
        latent_state = relay.var(f"itergp_latent_state_var_{_identifier_idx}", shape=latent_state_shape, dtype="bfloat16")
    else:
        latent_state = relay.var(f"itergp_latent_state_var_{_identifier_idx}", shape=latent_state_shape, dtype="float32") # TODO: add type inference
    
    current_idx = relay.var(f"itergp_cur_idx_{_identifier_idx}", shape=(3,), dtype="int32")
    _identifier_idx += 1 

    attrs = tvm.ir.make_node("DictAttrs", pool_type=pool_type, pool_size=pool_size,
                            latent_dim=latent_dim, stride=stride, latent_state_shape=latent_state_shape, bf16=bf16)

    return relay.Call(relay.op.get(op_name), [x, latent_state, current_idx], attrs)


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper


from tvm.topi import utils

T = tvm.tir

def u16tof32(v):
    uint32_v = v.astype("uint32")
    uint32_v = uint32_v << tvm.tir.const(16, "uint32")
    return T.reinterpret("float32", uint32_v)


def _bf16tof32(v):
    return u16tof32(T.reinterpret("uint16", v))


def f32tou16(v):
    uint32_v = T.reinterpret("uint32", v)
    rounding_bias = (uint32_v >> tvm.tir.const(16, "uint32")) & tvm.tir.const(1, "uint32")
    rounding_bias += tvm.tir.const(0x7FFF, "uint32")
    uint32_v = uint32_v + rounding_bias
    return (uint32_v >> tvm.tir.const(16, "uint32")).astype("uint16")


def _f32tobf16(v):
    return T.reinterpret("bfloat16", f32tou16(v))

import tvm.tir.op as tir_op

@relay.op.op.register_compute(op_name)
def _compute(attrs, inputs, output_type):
    print("We are now at iterative_global_pool_compute")  
    
    #current_idx[1,2,3] = [_num_cur_pool_filled, _num_cur_cell_filled, _cur_cell_idx]

    data, latent_state, current_idx = inputs

    pool_type = attrs["pool_type"]
    pool_size = attrs["pool_size"]
    latent_dim = attrs["latent_dim"]
    stride = attrs["stride"]
    latent_state_shape = attrs["latent_state_shape"]
    bf16 = attrs["bf16"]
    num_of_latent_state = latent_state_shape[-1]


    if not bf16:
        f32tobf16 = lambda x : x
        bf16tof32 = lambda x : x
    else:
        f32tobf16 = _f32tobf16
        bf16tof32 = _bf16tof32

    if pool_type == "Avg":
        cell_pool_func = lambda a, b: a + (b / pool_size)
        output_func = lambda a: tir_op.sum(a)
    elif pool_type == "Max":
        cell_pool_func = lambda a, b: tir_op.max(a, b)
        output_func = lambda a: tir_op.max(a)
    else:
        raise NotImplementedError(f"IterativeGlobalPool type: {pool_type} is not supported.")

    def gen_ib_4d(data_buf, buffer_var_buf, current_idx_buf, out_buf):
        ib = tvm.tir.ir_builder.create()
        data = ib.buffer_ptr(data_buf)
        buffer = ib.buffer_ptr(buffer_var_buf)
        cur_idx = ib.buffer_ptr(current_idx_buf)
        _num_cur_pool_filled, _num_cur_cell_filled, _cur_cell_idx = cur_idx[0], cur_idx[1], cur_idx[2],
        out = ib.buffer_ptr(out_buf)

        # Copy new data to buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  latent_state_shape[1], "i") as i:
                with ib.for_range(0, latent_state_shape[2], "j") as j:
                    buffer[n , i , j , _cur_cell_idx] = f32tobf16(cell_pool_func(bf16tof32(buffer[n , i , j , _cur_cell_idx]), data[n, i , j , 0]))

        _num_cur_cell_filled += 1
        with ib.if_scope(_num_cur_cell_filled >= stride):
            _num_cur_cell_filled = 0
            _cur_cell_idx += 1
            _cur_cell_idx %= num_of_latent_state
        
        _num_cur_pool_filled += 1
        with ib.if_scope(_num_cur_pool_filled >= pool_size):
            # copy to Output buffer
            with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
                with ib.for_range(0,  latent_state_shape[1], "i") as i:
                    with ib.for_range(0, latent_state_shape[2], "j") as j:
                            out[n , i , j , 0] = bf16tof32(output_func(buffer[n, i , j]))
            _num_cur_pool_filled = _num_cur_pool_filled - stride

        with ib.else_scope():
            # skip output
            ib.emit(tvm.tir.ret(-1))
        return ib.get()

    def gen_ib_3d(data_buf, buffer_var_buf, current_idx_buf, out_buf):
        ib = tvm.tir.ir_builder.create()
        data = ib.buffer_ptr(data_buf)
        buffer = ib.buffer_ptr(buffer_var_buf)
        cur_idx = ib.buffer_ptr(current_idx_buf)
        _num_cur_pool_filled, _num_cur_cell_filled, _cur_cell_idx = cur_idx[0], cur_idx[1], cur_idx[2],
        out = ib.buffer_ptr(out_buf)

        # Copy new data to buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  latent_state_shape[1], "i") as i:
                buffer[n , i , _cur_cell_idx] = f32tobf16(cell_pool_func(bf16tof32(buffer[n , i , _cur_cell_idx]), data[n, i , 0]))

        _num_cur_cell_filled += 1
        with ib.if_scope(_num_cur_cell_filled >= stride):
            _num_cur_cell_filled = 0
            _cur_cell_idx += 1
            _cur_cell_idx %= num_of_latent_state
        
        _num_cur_pool_filled += 1
        with ib.if_scope(_num_cur_pool_filled >= pool_size):
            # copy to Output buffer
            with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
                with ib.for_range(0,  latent_state_shape[1], "i") as i:
                        out[n , i ,0] = bf16tof32(output_func(buffer[n, i]))
            _num_cur_pool_filled = _num_cur_pool_filled - stride

        with ib.else_scope():
            # skip output
            ib.emit(tvm.tir.ret(-1))
        return ib.get()

    def gen_ib_2d(data_buf, buffer_var_buf, current_idx_buf, out_buf):
        ib = tvm.tir.ir_builder.create()
        data = ib.buffer_ptr(data_buf)
        buffer = ib.buffer_ptr(buffer_var_buf)
        cur_idx = ib.buffer_ptr(current_idx_buf)
        _num_cur_pool_filled, _num_cur_cell_filled, _cur_cell_idx = cur_idx[0], cur_idx[1], cur_idx[2],
        out = ib.buffer_ptr(out_buf)

        # Copy new data to buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            buffer[n , _cur_cell_idx] = f32tobf16(cell_pool_func(bf16tof32(buffer[n , _cur_cell_idx]), data[n, 0]))

        _num_cur_cell_filled += 1
        with ib.if_scope(_num_cur_cell_filled >= stride):
            _num_cur_cell_filled = 0
            _cur_cell_idx += 1
            _cur_cell_idx %= num_of_latent_state
        
        _num_cur_pool_filled += 1
        with ib.if_scope(_num_cur_pool_filled >= pool_size):
            # copy to Output buffer
            with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
                out[n , 0] = bf16tof32(output_func(buffer[n]))
            _num_cur_pool_filled = _num_cur_pool_filled - stride

        with ib.else_scope():
            # skip output
            ib.emit(tvm.tir.ret(-1))
        return ib.get()
    
    if len(latent_state_shape) == 4:
        out_ib = [tvm.te.extern(output_type.shape, [data, latent_state, current_idx],
                lambda ins, outs: gen_ib_4d(ins[0], ins[1], ins[2], outs[0]),
                name=op_name + "_compute.generic", 
                dtype=output_type.dtype,
                )]
    elif len(latent_state_shape) == 3:
        out_ib = [tvm.te.extern(output_type.shape, [data, latent_state, current_idx],
                lambda ins, outs: gen_ib_3d(ins[0], ins[1], ins[2], outs[0]),
                name=op_name + "_compute.generic", 
                dtype=output_type.dtype,
                )]
    elif len(latent_state_shape) == 2:
        out_ib = [tvm.te.extern(output_type.shape, [data, latent_state, current_idx],
                lambda ins, outs: gen_ib_2d(ins[0], ins[1], ins[2], outs[0]),
                name=op_name + "_compute.generic", 
                dtype=output_type.dtype,
                )]
    return out_ib

@override_native_generic_func(op_name + "_strategy")
def _strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        _compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        # None,
        name=op_name + ".generic",
    )
    return strategy
_op.register_strategy(op_name, _strategy)

