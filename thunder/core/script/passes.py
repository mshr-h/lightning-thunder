import dis
import inspect

import opcode
import torch  # # aehem.

import thunder

from .frontend import acquire_method, make_single_return, make_ssa
from .graph import Block, Node, PhiValue, replace_values


def specify_inputs(gr, inps):
    inp_map = {p: v for p, v in zip(gr.local_variables_at_start, inps)}
    replace_values(gr, inp_map)


def split_block(gr, bl, n):
    # The admin involved:
    # - create a new "bottom block", the input block is the "top block"
    # - split the .nodes
    # - block_inputs of the top block and block_outputs of the bottom are the original
    #   block_inputs and block_outputs
    # - scan all the node inputs and block_outputs of the lower part to see
    #   which need to be block_inputs of the lower block and thus outputs of the top one
    # - define outputs of the "top block" to be the required inputs
    # - add the input PhiValues and replace the outputs of the top block with them in the
    #   uses in the bottom block
    # - add unconditional jump from top to bottom part

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
    nbl.block_outputs = bl.block_outputs
    bl.block_outputs = set()
    nbl.block_inputs = []

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
    bl_jump_node = Node(i=jump_ins, inputs=[], outputs=[])
    bl_jump_node.jump_targets = [((0, 0), nbl)]
    bl.nodes.append(bl_jump_node)
    nbl.jump_sources.append(bl_jump_node)
    gr.blocks.insert(i + 1, nbl)

    nbl_block_inputs = {}
    potential_bl_outputs = {i for i in bl.block_inputs}
    for n in bl.nodes:
        for o in n.outputs:
            potential_bl_outputs.add(o)
    for i in bl.block_inputs:
        potential_bl_outputs.add(i)

    def get_or_create_phi(v):
        phi_value = nbl_block_inputs.get(v)
        if phi_value is None:
            phi_value = PhiValue([v], [bl_jump_node], nbl)
            nbl.block_inputs.append(phi_value)
        return phi_value

    for n in nbl.nodes:
        for idx_i, i in enumerate(n.inputs):
            if i in potential_bl_outputs:
                n.inputs[idx_i] = get_or_create_phi(i)
                bl.block_outputs.add(i)
        # for inplace ops, we also check the outputs (e.g. FOR_ITER)
        for idx_o, o in enumerate(n.outputs):
            if o in potential_bl_outputs:
                o.outputs[idx_o] = get_or_create_phi(o)
                bl.block_outputs.add(o)

    bl.block_outputs.update(nbl.block_outputs & potential_bl_outputs)
    nbl.block_outputs = {(get_or_create_phi(o) if o in potential_bl_outputs else o) for o in nbl.block_outputs}

    return nbl


def inline_method_call(gr, n):  # criterion?
    found_block = False
    for i_bl, bl in enumerate(gr.blocks):
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
    """replaces calls to torch.foo functions with calls into thunder's torch
    language."""
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
