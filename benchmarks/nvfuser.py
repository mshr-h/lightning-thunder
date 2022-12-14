import argparse
import time
from functools import partial, reduce
import warnings
import operator
import math
import multiprocessing
from multiprocessing import Process
import os

import thunder
import thunder.core.lang as tlang
import thunder.langs.torch as ttorch

import torch
from torch.testing import make_tensor
from torch.utils.benchmark import Timer

# This file contains custom nvFuser-related benchmarks. 

def median(l):
    if len(l) == 0:
        raise ValueError("Empty lists have no median value!")

    s = sorted(l)

    if len(s) % 2 == 1:
        return s[len(s) // 2]

    return (s[len(s) // 2] + s[len(s) // 2 - 1]) / 2

def time_ns(fn, gen, *, warmup_iters=5, iters=10):
    total_time_ns = 0
    elapsed = []

    def _helper():
        args, kwargs = gen()
        torch.cuda.synchronize()
        start = time.time_ns()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time_ns()
        return end - start, result

    # First run
    initial_time, initial_result = _helper()

    for _ in range(warmup_iters):
        _helper()

    for _ in range(iters):
        t, result = _helper()
        elapsed.append(t)

    # Computes statistics
    avg = reduce(operator.add, elapsed, 0) / iters
    m = median(elapsed)
    
    stats = {
        "warmup_iters": warmup_iters,
        "iters": iters,
        "initial": initial_time,
        "initial_result": initial_result,
        "average": avg,
        "median" : m,
        "final": elapsed[-1],
        "final_result": result,
    }

    return stats

# TODO: do we want to use timer?
def _timer(fn, *args, iters=10, **kwargs):
    """
    A wrapper around PyTorch's timer. Returns a Measurement object, described here:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/benchmark/utils/common.py
    """
    kwargs = kwargs or {}
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    fn_call = "fn(*args, **kwargs)"

    timer = Timer(stmt=f"{fn_call}", globals=env)
    tt = timer.timeit(iters)

    return tt

def make_traced(fn):
    return thunder.make_traced(fn, executor="nvfuser", _info=True)

def _prettyprint_thunder_nvfuser_stats(stats):
    def _print_breakdown(first):
        s = "first" if first else "final"
        ns = stats['initial'] if first else stats['final']
        print(f"The {s} iteration took {ns}ns, and can be broken down as...")

        result, meta = stats['initial_result'] if first else stats['final_result']

        a_time = meta['acquisition_time']
        t_time = meta['translation_time']
        e_time = meta['execution_time']
        print(f"{a_time}, {round(a_time / ns, 2)}% of the time, was spent in program acquisition")
        print(f"{t_time}ns, {round(t_time / ns, 2)}% of the time, was spent translating the program to a fusion definition")
        print(f"{e_time}ns, {round(e_time / ns, 2)}% of the time, was spent asking nvFuser to start executing")

        accounted_time = a_time + t_time + e_time
        unaccounted_time = ns - accounted_time
        print(f"{unaccounted_time}ns, {round(unaccounted_time / ns, 2)}% of the time, is unaccounted for, but is probably how long the kernels took to execute.")

    print("Thunder+nvFuser results:")
    print(f"The median time of {stats['iters']} post-warmup iterations was {stats['median']}")
    _print_breakdown(True)
    _print_breakdown(False)

def _prettyprint_stats(name, stats):
    print(f"{name} results:")
    print(f"The median time of {stats['iters']} post-warmup iterations was {stats['median']}")
    print(f"The initial interation took {stats['initial']}ns")
    print(f"The final interation took {stats['final']}ns")

def _compare_stats(name_a, stats_a, name_b, stats_b):
    a_initial = stats_a['initial']
    b_initial = stats_b['initial']

    if a_initial < b_initial:
        print(f"{name_a} was initially faster than {name_b}, taking only {round(a_initial/b_initial, 2)}% of the time")
    else:
        print(f"{name_b} was initially faster than {name_a}, taking only {round(b_initial/a_initial, 2)}% of the time")

    a_final = stats_a['final']
    b_final = stats_b['final']

    if a_final < b_final:
        print(f"{name_a} was finally faster than {name_b}, taking only {round(a_final/b_final, 2)}% of the time")
    else:
        print(f"{name_b} was finally faster than {name_a}, taking only {round(b_final/a_final, 2)}% of the time")

def _benchmark(name, *, gen, iters, thunder_fn, other_name, other_fn):
    print(f"Benchmark: {name}")
    thunder_stats = time_ns(thunder_fn, gen)
    _prettyprint_thunder_nvfuser_stats(thunder_stats)

    other_stats = time_ns(other_fn, gen)
    _prettyprint_stats(other_name, other_stats)
    _compare_stats("Thunder + nvFuser", thunder_stats, other_name, other_stats)

#
# Elementwise binary benchmarks
#

def _add_nvfuser_vs_dynamo_factory(shape, *, iters, make_arg, gen=None):
    if gen is None:
        def gen():
            a = make_arg(shape)
            b = make_arg(shape)
            return (a, b), {}

    # LABEL: Thunder
    thunder_fn = make_traced(tlang.add)
    
    # # LABEL: Dynamo
    def _add(a, b):
        return a + b

    shape_str = 'x'.join(str(l) for l in shape)
    dynamo_fn = torch.compile(_add)
    _benchmark(f"add_{shape_str}", gen=gen, iters=iters, thunder_fn=thunder_fn, other_name='dynamo', other_fn=dynamo_fn)

def add_64x64(iters, make_arg):
    _add_nvfuser_vs_dynamo_factory((64, 64), iters=iters, make_arg=make_arg)

def add_1024x1024(iters, make_arg):
    _add_nvfuser_vs_dynamo_factory((1024, 1024), iters=iters, make_arg=make_arg)

def add_4096x4(iters, make_arg):
    _add_nvfuser_vs_dynamo_factory((4096, 4), iters=iters, make_arg=make_arg)

def add_4x4096(iters, make_arg):
    _add_nvfuser_vs_dynamo_factory((4, 4096), iters=iters, make_arg=make_arg)

def _add_contiguous_tranposed_nvfuser_vs_dynamo_factory(shape, *, iters, make_arg):
    def gen():
        a = make_arg(shape)
        b = make_arg(shape).T
        return (a, b), {}

    # LABEL: Thunder
    thunder_fn = make_traced(tlang.add)
    
    # # LABEL: Dynamo
    def _add(a, b):
        return a + b

    shape_str = 'x'.join(str(l) for l in shape)
    dynamo_fn = torch.compile(_add)
    _benchmark(f"add_{shape_str}_contiguous_tranposed", gen=gen, iters=iters, thunder_fn=thunder_fn, other_name='dynamo', other_fn=dynamo_fn)

def add_1024x1024_contiguous_transposed(iters, make_arg):
    _add_contiguous_tranposed_nvfuser_vs_dynamo_factory((1024, 1024), iters=iters, make_arg=make_arg)

#
# Elementwise unary benchmarks
#

def _elementwise_unary_nvfuser_vs_dynamo_factory(shape, *, thunder_op, torch_op, iters, make_arg):
    def gen():
        return (make_arg(shape),), {}

    # LABEL: Thunder
    thunder_fn = make_traced(thunder_op)

    # LABEL: dynamo
    def _foo(a):
        return torch_op(a)
    dynamo_fn = torch.compile(_foo)

    shape_str = 'x'.join(str(l) for l in shape)
    name = f"{torch_op.__name__}{shape_str}"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_fn, other_name='dynamo', other_fn=dynamo_fn)

def abs_64x64(iters, make_arg):
    _elementwise_unary_nvfuser_vs_dynamo_factory((64, 64), thunder_op=tlang.abs, torch_op=torch.abs, iters=iters, make_arg=make_arg)

#
# Reduction benchmarks
#

def _all_reduce_nvfuser_vs_dynamo_factory(shape, *, thunder_op, torch_op, iters, make_arg):
    def gen():
        return (make_arg(shape),), {}

    # LABEL: Thunder
    thunder_fn = make_traced(thunder_op)

    # LABEL: dynamo
    def _foo(a):
        return torch_op(a)
    dynamo_fn = torch.compile(_foo)

    shape_str = 'x'.join(str(l) for l in shape)
    name = f"{torch_op.__name__}{shape_str}_all_reduce"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_fn, other_name='dynamo', other_fn=dynamo_fn)

def var_1024x1024_all_reduce(iters, make_arg):
    _all_reduce_nvfuser_vs_dynamo_factory((1024, 1024), thunder_op=ttorch.var, torch_op=torch.var, iters=iters, make_arg=make_arg)


benchmarks = {
    # Elementwise Binary benchmarks
    "add_64x64": add_64x64,
    "add_1024x1024": add_1024x1024,
    "add_4096x4": add_4096x4,
    "add_4x4096": add_4x4096,
    "add_1024x1024_contiguous_transposed": add_1024x1024_contiguous_transposed,
    # Elementise Unary benchmarks
    "abs_64x64": abs_64x64,
    # Reduction benchmarks
    "var_1024x1024_all_reduce": var_1024x1024_all_reduce,
}

def _run_benchmark(benchmark_fn, *args):
    p = Process(target=benchmark_fn, args=args)
    p.start()
    p.join()

# TODO: allow specifying iters, dtype, benchmarks
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", action="store", help="float32, int64, ...")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Acquires dtype (default is float32)
    dtype = thunder.float32
    if args.dtype is not None:
        dtype = getattr(thunder, args.dtype)
        if not thunder.dtypes.is_dtype(dtype):
            raise ValueError("Unknown dtype {args.dtype} specified!")

    # TODO: allow specifying a particular CUDA device
    device = 'cuda'

    iters = 10

    make_arg = partial(make_tensor, device=device, dtype=ttorch.torch_dtype(dtype))

    multiprocessing.set_start_method('spawn')

    # Ignores warnings during benchmarks
    # NOTE: dynamo will throw extraneous warnings
    os.environ["PYTHONWARNINGS"] = "ignore" 
    for k, v in benchmarks.items():
        _run_benchmark(v, iters, make_arg)


    
    



    