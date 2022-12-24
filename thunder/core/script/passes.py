import dis
import inspect
import opcode

from .frontend import acquire_method, make_single_return, make_ssa
from .graph import Block, Node, replace_values
import torch  ## aehem.
import thunder


def specify_inputs(gr, inps):
    inp_map = {p: v for p, v in zip(gr.local_variables_at_start, inps)}
    replace_values(gr, inp_map)


def split_block(gr, bl, n):
    i = 0
    while i < len(gr.blocks) and gr.blocks[i] is not bl:
        i += 1
    assert i < len(gr.blocks), "block not found"
    j = 0
    while j < len(bl.nodes) and bl.nodes[j] is not n:
        j += 1
    assert j < len(bl.nodes), "node not found"
    nbl = Block(is_ssa=True)
    nbl.nodes = bl.nodes[j:]
    del bl.nodes[j:]
    jump_ins = dis.Instruction(
        opname="JUMP_ABSOLUTE",
        opcode=opcode.opmap["JUMP_ABSOLUTE"],
        arg=None,
        argval=None,
        argrepr=None,
        offset=None,  # last_node_i.offset,
        starts_line=None,
        is_jump_target=False,
    )
    jump_node = Node(i=jump_ins, inputs=[], outputs=[])
    jump_node.jump_targets = [((0, 0), nbl)]
    bl.nodes.append(jump_node)
    nbl.jump_sources.append(jump_node)
    gr.blocks.insert(i + 1, nbl)
    return nbl


def inline_methode_call(gr, n):  # criterion?
    found_block = False
    for bl in gr.blocks:
        if n in bl.nodes:
            found_block = True
            break
    assert found_block
    if n.i.opname == "CALL_METHOD" and n.inputs[0].value is not None and not inspect.isbuiltin(n.inputs[0].value):
        nbl = split_block(gr, bl, n)
        assert nbl.nodes[0] == n
        del nbl.nodes[0]

        meth1 = n.inputs[0].value
        mod1 = n.inputs[1].value
        if isinstance(meth1, torch.nn.Module):  # when inlining works really well, we might switch to using __call__
            mod1 = meth1
            meth1 = meth1.forward
        gr1 = acquire_method(meth1, module=mod1, mro_klass=gr.mro_klass if mod1 == gr.module else None)
        make_ssa(gr1)
        make_single_return(gr1)

        # there should be exactly one
        (ret_bl,) = (bl for bl in gr1.blocks if len(bl.nodes) > 0 and bl.nodes[-1].i.opname == "RETURN_VALUE")

        if gr1.ismethod:
            specify_inputs(gr1, [n.inputs[1], *n.inputs[2:]])
        else:
            specify_inputs(gr1, n.inputs[2:])

        ret_node = ret_bl.nodes[-1]
        ret_node.i = dis.Instruction(
            opname="JUMP_ABSOLUTE",
            opcode=opcode.opmap["JUMP_ABSOLUTE"],
            arg=None,
            argval=None,
            argrepr=None,
            offset=ret_node.i.offset,
            starts_line=ret_node.i.starts_line,
            is_jump_target=ret_node.i.is_jump_target,
        )
        bl.nodes[-1].jump_targets = [((0, 0), gr1.blocks[0])]
        ret_node.jump_targets = [((0, 0), nbl)]
        # output values...
        rv = ret_node.inputs.pop()
        assert not ret_node.inputs
        (orv,) = n.outputs
        replace_values(gr, {orv: rv})
        gr.blocks[i_bl + 1 : i_bl + 1] = gr1.blocks


def torch_to_thunder(gr):
    """replaces calls to torch.foo functions with calls into thunder's torch language"""
    for bl in gr.blocks:
        for n in bl.nodes:
            for i in n.inputs:
                # todo: change name?, deeper nesting?
                if i.value == torch:
                    i.value = thunder.langs.torch
                if i.parent is not None and i.parent.value == torch:
                    i.parent.value = thunder.langs.torch
                    i.value = getattr(thunder.langs.torch, i.name)

                # replace other things by checking against torch module (make dict at startup?)
                n = getattr(i.value, "__name__", None)
                tf = None
                if n is not None:
                    tf = getattr(torch, n, None)
                if tf is not None and i.value == tf:
                    i.value = getattr(thunder.langs.torch, n)
                    i.is_global = False
                    i.is_const = True
