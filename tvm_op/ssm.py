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
op_name = "ssm"
relay.op.op.register("ssm")


# call default relation functions
def _rel(args, attrs):
    # func = attrs["relay_func"]
    # return relay.TupleType([relay.TensorType(attrs["output_shape"], args[1].dtype), relay.TensorType(func.body.checked_type.shape,func.body.checked_type.dtype)])
    return args[1]
_op.get(op_name).add_type_rel("SSMTypeRel", _rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
_op.register_stateful(op_name, True)

def ssm(x, num_of_latent_state, latent_dim, stride):
    latent_state_shape = tuple([*latent_dim, num_of_latent_state])
    latent_state = relay.var("ssm_latent_state_var", shape=latent_state_shape, dtype=x.dtype)
    current_idx = relay.var("ssm_cur_idx", shape=(1,), dtype="int32")

    attrs = tvm.ir.make_node("DictAttrs", num_of_latent_state=num_of_latent_state, 
                            latent_dim=latent_dim, stride=stride, latent_state_shape=latent_state_shape)

    return relay.Call(relay.op.get("ssm"), [x, latent_state, current_idx], attrs)


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper


from tvm.topi import utils

@relay.op.op.register_compute(op_name)
def _compute(attrs, inputs, output_type):
    print("We are now at cache_conv_input_compute")  
    
    data, latent_state, current_idx = inputs

    max_idx = attrs["max_idx"]
    buffer_shape = attrs["buffer_shape"]
    kernel_size = attrs["conv_kernel_size"]
    strides = attrs["conv_strides"]
    padding = attrs["conv_padding"]
    input_shape = attrs["conv_input_shape"]
    
    print("kernel_size:", kernel_size)
    print("input_shape:", input_shape)
    print("padding:", padding)
    print("strides:", strides)
    worker_id = get_fusion_worker_id()
    cur_op_num = get_fusion_op_num()
    print("worker_id:", worker_id, "op_num:", cur_op_num)

    def gen_ib(data_buf, buffer_var_buf, current_idx_buf, out_buf):
        ib = tvm.tir.ir_builder.create()
        ib.emit(invoke_c_macro(INSERT_STUB, worker_id, cur_op_num))
        ib.emit(invoke_c_macro(DECLEAR_EXTERN_WAIT_FOR_VAR, worker_id))
        ib.emit(invoke_c_macro(CHECK_IF_SKIP_COMPUTE, worker_id, cur_op_num, cur_op_num+1))
        data_w = data_buf.shape[3]
        data = ib.buffer_ptr(data_buf)
        buffer = ib.buffer_ptr(buffer_var_buf)
        cur_idx = ib.buffer_ptr(current_idx_buf)
        out = ib.buffer_ptr(out_buf)
        #shrift elements inside buffer
        # with ib.for_range(0 , buffer_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
        #     with ib.for_range(0,  buffer_shape[1], "i") as i:
        #         with ib.for_range(0, buffer_shape[2], "j") as j:
        #             with ib.for_range(0, buffer_shape[3] - data_w, "k") as k:
        #                 buffer[n , i , j , k] = buffer[n, i , j , k + data_w]

        # Copy new data to buffer
        with ib.for_range(0 , buffer_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  buffer_shape[1], "i") as i:
                with ib.for_range(0, buffer_shape[2], "j") as j:
                    with ib.for_range(0, data_w, "k") as k:
                        buffer[n , i , j , cur_idx[3] % (buffer_shape[3] - data_w + k)] = data[n, i , j , k]

        cur_idx[3] = cur_idx[3] + data_w

        with ib.if_scope(tvm.tir.all((cur_idx[3] + 1) >= kernel_size[1], ((cur_idx[3] + 1 - kernel_size[1]) % strides[1]) == 0)):
            # copy to Output buffer
            with ib.for_range(0 , buffer_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
                with ib.for_range(0,  buffer_shape[1], "i") as i:
                    with ib.for_range(0, buffer_shape[2], "j") as j:
                        with ib.for_range(0, buffer_shape[3], "k") as k:
                            out[n , i , j , k] = buffer[n, i , j , k]

        with ib.else_scope():
            # skip output
            ib.emit(tvm.tir.ret(-1))

        return ib.get()
    out_ib = [tvm.te.extern(output_type.shape, [data, buffer_var, current_idx],
              lambda ins, outs: gen_ib(ins[0], ins[1], ins[2], outs[0]),
            name=op + "_compute.generic", 
            dtype=output_type.dtype,
            )]
    return out_ib

@override_native_generic_func(op_name + "_strategy")
def _strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        _compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name=op_name + ".generic",
    )
    return strategy
_op.register_strategy(op_name, _strategy)

