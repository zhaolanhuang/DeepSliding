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
    # print("SSM Type REL:", args[1])
    return args[1]
_op.get(op_name).add_type_rel("SSMTypeRel", _rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
_op.register_stateful(op_name, True)

# Avoid name conflictsin C Code
_identifier_idx = 1

def ssm(x, num_of_latent_state, latent_dim, stride):
    global _identifier_idx
    latent_state_shape = tuple([*latent_dim, num_of_latent_state])
    latent_state = relay.var(f"ssm_latent_state_var_{_identifier_idx}", shape=latent_state_shape) # TODO: add type inference
    current_idx = relay.var(f"ssm_cur_idx_{_identifier_idx}", shape=(1,), dtype="int32")
    _identifier_idx += 1 

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
    print("We are now at ssm_compute")  
    
    data, latent_state, current_idx = inputs

    num_of_latent_state = attrs["num_of_latent_state"]
    latent_dim = attrs["latent_dim"]
    stride = attrs["stride"]
    latent_state_shape = attrs["latent_state_shape"]

    def gen_ib_4d(data_buf, buffer_var_buf, current_idx_buf, out_buf):
        ib = tvm.tir.ir_builder.create()
        data = ib.buffer_ptr(data_buf)
        buffer = ib.buffer_ptr(buffer_var_buf)
        cur_idx = ib.buffer_ptr(current_idx_buf)
        out = ib.buffer_ptr(out_buf)
        #shrift elements inside buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  latent_state_shape[1], "i") as i:
                with ib.for_range(0, latent_state_shape[2], "j") as j:
                    with ib.for_range(0, latent_state_shape[3] - 1, "k") as k:
                        buffer[n , i , j , k] = buffer[n, i , j , k + 1]
        # Copy new data to buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  latent_state_shape[1], "i") as i:
                with ib.for_range(0, latent_state_shape[2], "j") as j:
                    with ib.for_range(0, 1, "k") as k:
                        buffer[n , i , j , latent_state_shape[3] - 1 - k] = data[n, i , j , k]

        cur_idx[0] = cur_idx[0] + 1

        with ib.if_scope(cur_idx[0] >= num_of_latent_state):
            # copy to Output buffer
            with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
                with ib.for_range(0,  latent_state_shape[1], "i") as i:
                    with ib.for_range(0, latent_state_shape[2], "j") as j:
                        with ib.for_range(0, latent_state_shape[3], "k") as k:
                            out[n , i , j , k] = buffer[n, i , j , k]
            cur_idx[0] = cur_idx[0] - stride

        with ib.else_scope():
            # skip output
            ib.emit(tvm.tir.ret(-1))
        return ib.get()

    def gen_ib_3d(data_buf, buffer_var_buf, current_idx_buf, out_buf):
        ib = tvm.tir.ir_builder.create()
        data = ib.buffer_ptr(data_buf)
        buffer = ib.buffer_ptr(buffer_var_buf)
        cur_idx = ib.buffer_ptr(current_idx_buf)
        out = ib.buffer_ptr(out_buf)
        #shrift elements inside buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  latent_state_shape[1], "i") as i:
                with ib.for_range(0, latent_state_shape[2] - 1, "k") as k:
                    buffer[n , i, k] = buffer[n, i, k + 1]
        # Copy new data to buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  latent_state_shape[1], "i") as i:
                with ib.for_range(0, 1, "k") as k:
                    buffer[n , i , latent_state_shape[2] - 1 - k] = data[n, i , k]

        cur_idx[0] = cur_idx[0] + 1

        with ib.if_scope(cur_idx[0] >= num_of_latent_state):
            # copy to Output buffer
            with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
                with ib.for_range(0,  latent_state_shape[1], "i") as i:
                    with ib.for_range(0, latent_state_shape[2], "k") as k:
                        out[n , i , k] = buffer[n, i , k]
            cur_idx[0] = cur_idx[0] - stride

        with ib.else_scope():
            # skip output
            ib.emit(tvm.tir.ret(-1))
        return ib.get()

    def gen_ib_2d(data_buf, buffer_var_buf, current_idx_buf, out_buf):
        ib = tvm.tir.ir_builder.create()
        data = ib.buffer_ptr(data_buf)
        buffer = ib.buffer_ptr(buffer_var_buf)
        cur_idx = ib.buffer_ptr(current_idx_buf)
        out = ib.buffer_ptr(out_buf)
        #shrift elements inside buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0, latent_state_shape[1] - 1, "k") as k:
                buffer[n , k] = buffer[n, k + 1]
        # Copy new data to buffer
        with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0, 1, "k") as k:
                buffer[n , latent_state_shape[1] - 1 - k] = data[n, k]

        cur_idx[0] = cur_idx[0] + 1

        with ib.if_scope(cur_idx[0] >= num_of_latent_state):
            # copy to Output buffer
            with ib.for_range(0 , latent_state_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
                with ib.for_range(0, latent_state_shape[1], "k") as k:
                    out[n , k] = buffer[n, k]
            cur_idx[0] = cur_idx[0] - stride

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
        name=op_name + ".generic",
    )
    return strategy
_op.register_strategy(op_name, _strategy)

