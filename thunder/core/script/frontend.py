import collections
import dis
import inspect
import itertools
import sys

import opcode

from .graph import Block, Graph, Node, unify_values, Value
from .python_ir import stack_effect_detail


class Super:
    pass


class MROAwareObjectRef:
    def __init__(self, obj, start_klass=None):
        self.obj = obj
        self.start_klass = start_klass

    def __getattr__(self, name):
        # print("###", self.obj, self.start_klass, name)
        i = 0
        mro = inspect.getmro(self.obj.__class__)
        if self.start_klass is not None:
            while i < len(mro) and not mro[i] == self.start_klass:
                i += 1
            i += 1
        while i < len(mro) and not hasattr(mro[i], name):
            i += 1
        if i >= len(mro):
            raise AttributeError(f"{name} not a member")
        return getattr(mro[i], name)


def acquire_method(method, module=None, mro_klass=None, verbose=False):
    assert sys.version_info >= (3, 10) and sys.version_info < (3, 11)
    if verbose:
        print(inspect.getsource(method))
    sig = inspect.signature(method)
    if module is None and hasattr(method, "__self__"):
        module = method.__self__
    if mro_klass is None and module is not None:
        mro_klass = type(module)
    local_variables = []
    if inspect.ismethod(method):
        local_variables.append(Value(value=module, name=method.__code__.co_varnames[0]))
    for p in sig.parameters.values():
        assert (
            p.name == method.__code__.co_varnames[len(local_variables)]
        ), f"mismatch {p.name} {method.__code__.co_varnames[len(local_variables)]}"
        local_variables.append(Value(typ=p.annotation, name=p.name))
    ## KWARGS?!
    for i in enumerate(method.__code__.co_varnames, start=len(local_variables)):
        local_variables.append(None)

    # bound_args = [module.forward.__self__]
    bc = list(dis.get_instructions(method))
    if verbose:
        print(dis.dis(method))
    # Map offset_start -> Block
    block_0 = Block()
    block_0.jump_sources.append(None)
    blocks_to_process = collections.OrderedDict({0: block_0})
    blocks = {}

    def append_if_needed(offset_start, bl, jump_source):
        for other_offset_start, other_bl in itertools.chain(blocks_to_process.items(), blocks.items()):
            if other_offset_start == offset_start:
                ### take anything?
                print("#oldbl##", offset_start, jump_source, other_bl.jump_sources)
                other_bl.jump_sources.append(jump_source)
                return other_bl
        print("#newbl##", offset_start, jump_source, other_bl.jump_sources)
        blocks_to_process[offset_start] = bl
        bl.jump_sources.append(jump_source)
        return bl

    line_no = 0
    while blocks_to_process:
        offset_start, bl = blocks_to_process.popitem(last=False)
        blocks[offset_start] = bl

        ic = offset_start
        # module_ref = MROAwareObjectRef(module)
        done = False
        while not done:
            i = bc[ic]
            if i.starts_line is not None:
                line_no = i.starts_line
            n = Node(i=i, line_no=line_no)

            # need to handle branching instructions here
            if i.opname == "FOR_ITER":
                b1 = append_if_needed(offset_start=ic + 1 + i.arg, bl=Block(), jump_source=n)
                # try to do values here?
                n.jump_targets = [(stack_effect_detail(i.opname, i.arg, jump=True), b1)]
            elif i.opname in {"POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"}:
                done = True
                b1 = append_if_needed(offset_start=ic + 1, bl=Block(), jump_source=n)
                b2 = append_if_needed(offset_start=i.arg, bl=Block(), jump_source=n)
                n.jump_targets = [
                    (stack_effect_detail(i.opname, i.arg, jump=False), b1),
                    (stack_effect_detail(i.opname, i.arg, jump=True), b2),
                ]
            elif i.opname == "JUMP_FORWARD":
                done = True
                b1 = append_if_needed(offset_start=ic + 1 + i.arg, bl=Block(), jump_source=n)
                n.jump_targets = [(stack_effect_detail(i.opname, i.arg, jump=True), b1)]
            elif i.opname == "JUMP_ABSOLUTE":
                done = True
                b1 = append_if_needed(offset_start=i.arg, bl=Block(), jump_source=n)
                n.jump_targets = [(stack_effect_detail(i.opname, i.arg, jump=True), b1)]
            elif i.opname == "RETURN_VALUE":
                done = True
            else:
                if verbose:
                    print(i)
            bl.nodes.append(n)
            ic += 1
            if ic < len(bc) and bc[ic].is_jump_target:
                ### check if needed?
                if i.opname not in {
                    "RETURN_VALUE",
                    "JUMP_FORWARD",
                    "JUMP_ABSOLUTE",
                    "RAISE_VARARGS",
                    "POP_JUMP_IF_FALSE",
                    "POP_JUMP_IF_TRUE",
                }:
                    # should insert jump absolute instead...
                    jump_ins = dis.Instruction(
                        opname="JUMP_ABSOLUTE",
                        opcode=opcode.opmap["JUMP_ABSOLUTE"],
                        arg=None,
                        argval=None,
                        argrepr=None,
                        offset=None,
                        starts_line=None,
                        is_jump_target=False,
                    )
                    jump_node = Node(i=jump_ins, inputs=[], outputs=[])
                    bl.nodes.append(jump_node)
                    b1 = append_if_needed(offset_start=ic, bl=Block(), jump_source=jump_node)
                    jump_node.jump_targets = [
                        (
                            stack_effect_detail(jump_ins.opname, jump_ins.arg, jump=True),
                            b1,
                        )
                    ]
                done = True
    gr = Graph(list(blocks.values()))
    gr.local_variables_at_start = local_variables
    gr.method = method
    gr.module = module
    gr.mro_klass = mro_klass
    return gr


def make_ssa(gr, verbose=False):
    for bl in gr.blocks:
        for n in bl.nodes:
            n.block = bl
        bl.all_stacks_at_start = [None if js is not None else [] for js in bl.jump_sources]
        bl.all_local_variables_at_start = [
            None if js is not None else gr.local_variables_at_start[:] for js in bl.jump_sources
        ]
    blocks_to_do = set(gr.blocks)
    while blocks_to_do:
        one_block_done = False
        for bl in list(blocks_to_do):
            all_deps_done = not any(
                js.block in blocks_to_do for js in bl.jump_sources if js is not None and js.block is not bl
            )
            if all_deps_done:
                one_block_done = True
                blocks_to_do.remove(bl)

                jump_sources = bl.jump_sources
                print(
                    f"js {jump_sources} {[type(s) for s in bl.all_stacks_at_start]=}",
                    bl,
                )
                # TODO: We cannot currently support loops. :/
                stack = [unify_values(v, jump_sources, bl) for v in zip(*bl.all_stacks_at_start)]
                local_variables = [unify_values(v, jump_sources, bl) for v in zip(*bl.all_local_variables_at_start)]
                print("###lv1", local_variables)

                new_nodes = []
                for n_idx, n in enumerate(bl.nodes):
                    i = n.i
                    pop, push = stack_effect_detail(i.opname, i.arg)  ## jump?
                    inputs = stack[-pop:] if pop > 0 else []
                    n.inputs = inputs[:]
                    assert len(inputs) == pop
                    if i.opname == "LOAD_FAST":
                        outputs = [local_variables[i.arg]]
                    elif i.opname == "STORE_FAST":
                        outputs = []
                        (local_variables[i.arg],) = inputs  # set name?
                    elif i.opname == "DELETE_FAST":
                        outputs = []
                        local_variables[i.arg] = None
                    elif i.opname == "LOAD_GLOBAL":
                        if gr.method.__code__.co_names[i.arg] != "super":
                            if inspect.ismethod(gr.method):
                                func = gr.method.__func__
                            else:
                                func = gr.method
                            gn = gr.method.__code__.co_names[i.arg]
                            gv = func.__globals__[gn]
                            outputs = [Value(name=gn, value=gv, is_global=True)]
                        else:
                            outputs = [Value(name="super", value=Super())]
                    elif i.opname == "CALL_FUNCTION" and i.arg == 0 and isinstance(inputs[0].value, Super):
                        outputs = [Value(value=MROAwareObjectRef(gr.module, start_klass=gr.mro_klass))]
                        print("##super#", outputs)
                    elif i.opname == "LOAD_METHOD":  # also used for modules (callables)
                        (obj,) = inputs
                        mn = gr.method.__code__.co_names[i.arg]
                        m = Value(parent=obj, name=mn)
                        if obj.value is not None:
                            m.value = getattr(obj.value, mn)
                            m.typ = type(m.value)
                        # error case
                        # print("#lm###", type(m), type(obj), str(obj.value)[:100], m.value)
                        if isinstance(obj.value, MROAwareObjectRef):
                            print("...###", obj.value.start_klass)
                        #    obj = obj.obj
                        outputs = [m, obj]
                    elif i.opname == "LOAD_CONST":
                        outputs = [Value(value=gr.method.__code__.co_consts[i.arg], is_const=True)]
                    elif i.opname == "CALL_METHOD":
                        print(n.inputs[0])
                        outputs = [Value(n=n, nr=k) for k in range(push)]
                        new_nodes.append(n)
                    elif i.opname == "FOR_ITER":
                        # JUMP TARGETS
                        outputs = [Value(n=cur_instruction, name=".for_iter_item")]
                        new_nodes.append(n)
                    elif i.opname in {
                        "POP_JUMP_IF_FALSE",
                        "POP_JUMP_IF_TRUE",
                        "JUMP_FORWARD",
                        "JUMP_ABSOLUTE",
                    }:
                        new_nodes.append(n)
                        outputs = []
                    # elif i.opname == "JUMP_FORWARD":
                    # elif i.opname == "JUMP_ABSOLUTE":
                    elif i.opname == "RETURN_VALUE":
                        assert len(stack) == 1
                        new_nodes.append(n)
                        outputs = []
                    else:
                        if verbose:
                            print("unhandled", i)
                        outputs = [Value(n=n, nr=k) for k in range(push)]
                        new_nodes.append(n)
                    if n.jump_targets is not None:
                        for (j_pop, j_push), jt in n.jump_targets:
                            idx_jt = jt.jump_sources.index(n)
                            j_stack = stack[:]
                            if j_pop > 0:
                                j_stack = j_stack[:-pop]
                            if j_push > 0:
                                j_stack.extend(outputs[:j_push])
                            jt.all_stacks_at_start[idx_jt] = j_stack
                            jt.all_local_variables_at_start[idx_jt] = local_variables[:]

                    n.outputs = outputs
                    ol = len(stack)
                    print(ol, pop, push, i.opname)
                    if pop > 0:
                        stack = stack[:-pop]
                    stack.extend(outputs)
                    assert (i.opname == "JUMP_ABSOLUTE" and i.arg == None and len(stack) == ol) or (
                        len(stack) - ol == opcode.stack_effect(i.opcode, i.arg)
                    )
                if bl.continue_at is not None:
                    bl.continue_at.all_local_variables_at_start.append((n, local_variables[:]))
                bl.nodes = new_nodes
        assert one_block_done
    for bl in gr.blocks:
        del bl.all_local_variables_at_start
        del bl.all_stacks_at_start
        bl.is_ssa = True


def make_single_return(gr):
    bls = [b for b in gr.blocks if b.nodes[-1].i.opname == "RETURN_VALUE"]
    if len(bls) > 1:
        assert bls[-1].is_ssa
        ret_node = bls[-1].nodes[-1]
        if len(bls[-1].nodes) == 1:
            ret_bl = bls[-1]
        else:
            ret_bl = Block(is_ssa=True)
            ret_bl.nodes = [ret_node]
            gr.blocks.append(ret_bl)
        all_return_values = []
        for b in bls:
            if b != ret_bl:
                ## jump sources + unify!!!
                last_node_i = b.nodes[-1].i
                jump_ins = dis.Instruction(
                    opname="JUMP_ABSOLUTE",
                    opcode=opcode.opmap["JUMP_ABSOLUTE"],
                    arg=None,
                    argval=None,
                    argrepr=None,
                    offset=last_node_i.offset,
                    starts_line=None,
                    is_jump_target=last_node_i.is_jump_target,
                )
                jump_node = Node(i=jump_ins, inputs=[], outputs=[])
                jump_node.jump_targets = [((0, 0), ret_bl)]
                ret_bl.jump_sources.append(jump_node)
                all_return_values.append(b.nodes[-1].inputs)
                del b.nodes[-1]
                b.nodes.append(jump_node)
        ret_node.inputs = [unify_values(values, ret_bl.jump_sources, ret_bl) for values in zip(*all_return_values)]
    return gr
