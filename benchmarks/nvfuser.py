import argparse
import math
import multiprocessing
import operator
import os
import time
import warnings
from functools import partial, reduce
from multiprocessing import Process

import torch
from torch.testing import make_tensor
from torch.utils.benchmark import Timer

import thunder
import thunder.core.lang as tlang
import thunder.langs.torch as ttorch

# This file contains custom nvFuser-related benchmarks.


def median(l):
    if len(l) == 0:
        raise ValueError("Empty lists have no median value!")

    s = sorted(l)

    if len(s) % 2 == 1:
        return s[len(s) // 2]

    return (s[len(s) // 2] + s[len(s) // 2 - 1]) / 2


def time_ns(fn, gen, *, warmup_iters=5, iters=20):
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
        "median": m,
        "final": elapsed[-1],
        "final_result": result,
    }

    return stats


# TODO: consider refactoring this function with the above so they share more code
def time_thunder_ns(fn, gen, *, warmup_iters=5, iters=20):
    fn = thunder.make_traced(fn, executor="nvfuser", _info=True, _return_fusion=True)

    def _helper(fn_):
        args, kwargs = gen()
        torch.cuda.synchronize()
        start = time.time_ns()
        result = fn_(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time_ns()
        return end - start, result

    # NOTE: Initial run
    initial_time, initial_result = _helper(fn)
    result, meta, fusion = initial_result

    # NOTE: Warmups (a common benchmarking technique)
    for _ in range(warmup_iters):
        _helper(fn)

    # Acquires lazy Thunder stats
    elapsed = []
    for _ in range(iters):
        t, result = _helper(fn)
        elapsed.append(t)

    # Computes lazy statistics
    avg = reduce(operator.add, elapsed, 0) / iters
    m = median(elapsed)

    stats = {
        "warmup_iters": warmup_iters,
        "iters": iters,
        "initial": initial_time,
        "initial_result": initial_result,
        "lazy_average": avg,
        "lazy_median": m,
        "lazy_final": elapsed[-1],
        "lazy_final_result": result,
    }

    # Computes fusion results
    elapsed = []
    for _ in range(iters):
        t, result = _helper(fusion)
        elapsed.append(t)

    # Computes fusion statistics
    avg = reduce(operator.add, elapsed, 0) / iters
    m = median(elapsed)

    stats.update(
        {
            "fusion_average": avg,
            "fusion_median": m,
            "fusion_final": elapsed[-1],
            "fusion_final_result": result,
        }
    )

    return stats


def percent(numerator, denominator):
    return f"{round(numerator / denominator * 100, 2)}%"


def ns_to_us(ns):
    return round(ns / 1000, 2)


def _prettyprint_thunder_nvfuser_stats(stats):
    us = "\u03BCs"

    def _print_breakdown(s):
        ns = None
        meta = None
        if s == "initial":
            ns = stats["initial"]
            _, meta, _ = stats["initial_result"]
        elif s == "lazy":
            ns = stats["lazy_final"]
            _, meta, _ = stats["lazy_final_result"]

        iter_desc = None
        if s == "initial":
            iter_desc = "first"
        elif s == "lazy":
            iter_desc = "final lazy"

        print(f"The {iter_desc} iteration took {ns_to_us(ns)}{us}, and can be broken down as...")
        a_time = meta["acquisition_time"]
        t_time = meta["translation_time"]
        e_time = meta["invocation_time"]
        print(f"{ns_to_us(a_time)}{us}, {percent(a_time, ns)} of the time, was spent in program acquisition")
        print(
            f"{ns_to_us(t_time)}{us}, {percent(t_time, ns)} of the time, was spent translating the program to a fusion definition"
        )
        print(f"{ns_to_us(e_time)}{us}, {percent(e_time, ns)} of the time, was spent invoking nvFuser.execute()")

        accounted_time = a_time + t_time + e_time
        unaccounted_time = ns - accounted_time
        print(
            f"{ns_to_us(unaccounted_time)}{us}, {percent(unaccounted_time, ns)} of the time, is unaccounted for, but is probably how long the kernels took to execute."
        )

    print("Thunder+nvFuser results:")
    print(f"The median time of {stats['iters']} lazy post-warmup iterations was {ns_to_us(stats['lazy_median'])}{us}")
    print(
        f"The median time of {stats['iters']} fused post-warmup iterations was {ns_to_us(stats['fusion_median'])}{us}"
    )
    _print_breakdown("initial")
    _print_breakdown("lazy")


def _prettyprint_stats(name, stats):
    us = "\u03BCs"

    print(f"{name} results:")
    print(f"The median time of {stats['iters']} post-warmup iterations was {ns_to_us(stats['median'])}{us}")
    print(f"The initial iteration took {ns_to_us(stats['initial'])}{us}")
    print(f"The final iteration took {ns_to_us(stats['final'])}{us}")


def _compare_stats(thunder_name, thunder_stats, name_b, stats_b):
    thunder_initial = thunder_stats["initial"]
    other_initial = stats_b["initial"]

    print(f"Results of comparing Thunder and {name_b}:")
    if thunder_initial < other_initial:
        print(
            f"{thunder_name} was initially faster than {name_b}, taking only {percent(thunder_initial, other_initial)} of the time"
        )
    else:
        print(
            f"{name_b} was initially faster than {thunder_name}, taking only {percent(other_initial, thunder_initial)} of the time"
        )

    thunder_lazy_median = thunder_stats["lazy_median"]
    other_median = stats_b["median"]
    name = f"Lazy {thunder_name}"

    if thunder_lazy_median < other_median:
        print(
            f"{name} was faster post-warmup than {name_b}, taking only {percent(thunder_lazy_median, other_median)} of the time"
        )
    else:
        print(
            f"{name_b} was faster post-warmup than {name}, taking only {percent(other_median, thunder_lazy_median)} of the time"
        )

    thunder_fused_median = thunder_stats["fusion_median"]
    name = f"Fused {thunder_name}"

    if thunder_fused_median < other_median:
        print(
            f"{name} was faster post-warmup than {name_b}, taking only {percent(thunder_fused_median, other_median)} of the time"
        )
    else:
        print(
            f"{name_b} was faster post-warmup than {name}, taking only {percent(other_median, thunder_fused_median)} of the time"
        )


def _benchmark(name, *, gen, iters, thunder_fn, other_name, other_fn):
    print(f"Benchmark: {name}")

    thunder_stats = time_thunder_ns(thunder_fn, gen, iters=iters)
    _prettyprint_thunder_nvfuser_stats(thunder_stats)

    other_stats = time_ns(other_fn, gen, iters=iters)
    _prettyprint_stats(other_name, other_stats)

    _compare_stats("Thunder + nvFuser", thunder_stats, other_name, other_stats)


#
# Elementwise binary benchmarks
#


def _add_nvfuser_vs_pt2_factory(shape, *, iters, make_arg):
    def gen():
        a = make_arg(shape)
        b = make_arg(shape)
        return (a, b), {}

    # Constructs pt2 function
    def _add(a, b):
        return a + b

    pt2_fn = torch.compile(_add)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(f"add_{shape_str}", gen=gen, iters=iters, thunder_fn=tlang.add, other_name="pt2", other_fn=pt2_fn)


def add_64x64(iters, make_arg):
    _add_nvfuser_vs_pt2_factory((64, 64), iters=iters, make_arg=make_arg)


def add_kwargs_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        a = make_arg(shape)
        b = make_arg(shape)
        return (), {"a": a, "b": b}

    # Constructs pt2 function
    def _add(a, b):
        return a + b

    pt2_fn = torch.compile(_add)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(f"add_kwargs_{shape_str}", gen=gen, iters=iters, thunder_fn=tlang.add, other_name="pt2", other_fn=pt2_fn)


def add_1024x1024(iters, make_arg):
    _add_nvfuser_vs_pt2_factory((1024, 1024), iters=iters, make_arg=make_arg)


def add_4096x4(iters, make_arg):
    _add_nvfuser_vs_pt2_factory((4096, 4), iters=iters, make_arg=make_arg)


def add_4x4096(iters, make_arg):
    _add_nvfuser_vs_pt2_factory((4, 4096), iters=iters, make_arg=make_arg)


def add_dozen_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        args = []
        for _ in range(12):
            args.append(make_arg(shape))
        return tuple(args), {}

    def _add_dozen(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11):
        return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11

    pt2_fn = torch.compile(_add_dozen)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(f"add_dozen_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_dozen, other_name="pt2", other_fn=pt2_fn)


def add_hundred_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        args = []
        for _ in range(100):
            args.append(make_arg(shape))
        return tuple(args), {}

    # fmt: off
    def _add_hundred(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99):
        return a0 + a1 + a2 + a3 + a4 + a5+ a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31 + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39 + a40 + a41 + a42 + a43 + a44 + a45 + a46 + a47 + a48 + a49 + a50 + a51 + a52 + a53 + a54 + a55 + a56 + a57 + a58 + a59 + a60 + a61 + a62 + a63 + a64 + a65 + a66 + a67 + a68 + a69 + a70 + a71 + a72 + a73 + a74 + a75 + a76 + a77 + a78 + a79 + a80 + a81 + a82 + a83 + a84 + a85 + a86 + a87 + a88 + a89 + a90 + a91 + a92 + a93 + a94 + a95 + a96 + a97 + a98 + a99
    # fmt: on

    pt2_fn = torch.compile(_add_hundred)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(
        f"add_hundred_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_hundred, other_name="pt2", other_fn=pt2_fn
    )


def add_dozen_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        args = []
        for _ in range(12):
            args.append(make_arg(shape))
        return tuple(args), {}

    def _add_dozen(*args):
        cur = args[0]
        for a in args[1:]:
            cur = cur + a
        return cur

    pt2_fn = torch.compile(_add_dozen)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(f"add_dozen_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_dozen, other_name="pt2", other_fn=pt2_fn)


def add_stack100_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        a = make_arg(shape)
        b = make_arg(shape)
        return (a, b), {}

    def _add_stack100(a, b):
        cur = a
        for _ in range(100):
            cur = cur + b

        return cur

    pt2_fn = torch.compile(_add_stack100)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(
        f"add_stack100_{shape_str}",
        gen=gen,
        iters=iters,
        thunder_fn=_add_stack100,
        other_name="pt2",
        other_fn=pt2_fn,
    )


def add_stack1000_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        a = make_arg(shape)
        b = make_arg(shape)
        return (a, b), {}

    def _add_stack1000(a, b):
        cur = a
        for _ in range(1000):
            cur = cur + b

        return cur

    pt2_fn = torch.compile(_add_stack1000)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(
        f"add_stack1000_{shape_str}",
        gen=gen,
        iters=iters,
        thunder_fn=_add_stack1000,
        other_name="pt2",
        other_fn=pt2_fn,
    )


def _add_contiguous_transposed_nvfuser_vs_pt2_factory(shape, *, iters, make_arg):
    def gen():
        a = make_arg(shape)
        b = make_arg(shape).T
        return (a, b), {}

    # Makes PyTorch2 function
    def _add(a, b):
        return a + b

    pt2_fn = torch.compile(_add)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(
        f"add_{shape_str}_contiguous_transposed",
        gen=gen,
        iters=iters,
        thunder_fn=tlang.add,
        other_name="pt2",
        other_fn=pt2_fn,
    )


def add_1024x1024_contiguous_transposed(iters, make_arg):
    _add_contiguous_transposed_nvfuser_vs_pt2_factory((1024, 1024), iters=iters, make_arg=make_arg)


#
# Elementwise unary benchmarks
#


def _elementwise_unary_nvfuser_vs_pt2_factory(shape, *, thunder_op, torch_op, iters, make_arg):
    def gen():
        return (make_arg(shape),), {}

    # Makes pt2 function
    def _foo(a):
        return torch_op(a)

    pt2_fn = torch.compile(_foo)

    shape_str = "x".join(str(l) for l in shape)
    name = f"{torch_op.__name__}{shape_str}"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_op, other_name="pt2", other_fn=pt2_fn)


def abs_64x64(iters, make_arg):
    _elementwise_unary_nvfuser_vs_pt2_factory(
        (64, 64), thunder_op=tlang.abs, torch_op=torch.abs, iters=iters, make_arg=make_arg
    )


#
# Reduction benchmarks
#


def _all_reduce_nvfuser_vs_pt2_factory(shape, *, thunder_op, torch_op, iters, make_arg):
    def gen():
        return (make_arg(shape),), {}

    # Makes pt2 function
    def _foo(a):
        return torch_op(a)

    pt2_fn = torch.compile(_foo)

    shape_str = "x".join(str(l) for l in shape)
    name = f"{torch_op.__name__}{shape_str}_all_reduce"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_op, other_name="pt2", other_fn=pt2_fn)


def var_1024x1024_all_reduce(iters, make_arg):
    _all_reduce_nvfuser_vs_pt2_factory(
        (1024, 1024), thunder_op=ttorch.var, torch_op=torch.var, iters=iters, make_arg=make_arg
    )


def simple_number_conditional(iters, make_arg):
    shape = (64, 64)

    def gen():
        return (make_arg(shape), make_arg(shape), 2), {}

    def foo(alpha, beta, n):
        if n < 0:
            result = alpha - beta
        else:
            result = alpha + beta

        return alpha, result

    pt2_fn = torch.compile(foo)

    name = f"simple_number_conditional"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=foo, other_name="pt2", other_fn=pt2_fn)


def simple_kwarg_conditional(iters, make_arg):
    shape = (64, 64)

    def gen():
        return (make_arg(shape), make_arg(shape)), {"n": 2}

    def foo(alpha, beta, n):
        if n < 0:
            result = alpha - beta
        else:
            result = alpha + beta

        return alpha, result

    pt2_fn = torch.compile(foo)

    name = f"simple_kwarg_conditional"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=foo, other_name="pt2", other_fn=pt2_fn)


#
# nanoGPT benchmarks
#
# TODO: maybe put these in their own file?


def _nanogpt_new_gelu_vs_pt2_factory(shape, *, iters, make_arg):
    def gen():
        a = make_arg(shape)
        return (a,), {}

    def new_gelu(a):
        return 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * torch.pow(a, 3.0))))

    def new_gelu_thunder(a):
        return 0.5 * a * (1.0 + ttorch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * ttorch.pow(a, 3.0))))

    pt2_fn = torch.compile(new_gelu)

    shape_str = "x".join(str(l) for l in shape)
    name = f"nanogpt_gelu_{shape_str}"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=new_gelu_thunder, other_name="pt2", other_fn=pt2_fn)


def nanogpt_new_gelu_64x64(iters, make_arg):
    _nanogpt_new_gelu_vs_pt2_factory((64, 64), iters=iters, make_arg=make_arg)


def nanogpt_new_gelu_512x512(iters, make_arg):
    _nanogpt_new_gelu_vs_pt2_factory((512, 512), iters=iters, make_arg=make_arg)


def nanogpt_new_gelu_1024x1024(iters, make_arg):
    _nanogpt_new_gelu_vs_pt2_factory((1024, 1024), iters=iters, make_arg=make_arg)


benchmarks = {
    # Elementwise Binary benchmarks
    "add_64x64": add_64x64,
    "add_kwargs_64x64": add_kwargs_64x64,
    "add_dozen_64x64": add_dozen_64x64,
    "add_hundred_64x64": add_hundred_64x64,
    "add_stack100_64x64": add_stack100_64x64,
    "add_stack1000_64x64": add_stack1000_64x64,
    "add_dozen_64x64": add_dozen_64x64,  # Requires supporting *args
    "add_1024x1024": add_1024x1024,
    "add_4096x4": add_4096x4,
    "add_4x4096": add_4x4096,
    "add_1024x1024_contiguous_transposed": add_1024x1024_contiguous_transposed,
    # Elementwise Unary benchmarks
    "abs_64x64": abs_64x64,
    # Reduction benchmarks
    "var_1024x1024_all_reduce": var_1024x1024_all_reduce,  # Requires supporting sequence proxies
    # Control flow benchmarks
    "simple_number_conditional": simple_number_conditional,
    "simple_kwarg_conditional": simple_kwarg_conditional,
    # Network snippet benchmarks
    "nanogpt_new_gelu_64x64": nanogpt_new_gelu_64x64,
    "nanogpt_new_gelu_512x512": nanogpt_new_gelu_512x512,
    "nanogpt_new_gelu_1024x1024": nanogpt_new_gelu_1024x1024,
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
    device = "cuda"

    iters = 20

    make_arg = partial(make_tensor, device=device, dtype=ttorch.torch_dtype(dtype))

    multiprocessing.set_start_method("spawn")

    # Ignores warnings during benchmarks
    # NOTE: setting this environment variable effective sets
    #   warnings.simplewarningsfilter('ignore') in each (sub)process
    # NOTE: pt2 will throw extraneous warnings
    os.environ["PYTHONWARNINGS"] = "ignore"
    for k, v in benchmarks.items():
        _run_benchmark(v, iters, make_arg)
