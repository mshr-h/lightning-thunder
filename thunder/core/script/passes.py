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


def find_method_through_phi_parent(fn_value):
    # for inlining, we need to (reverse) traverse PhiValues and attribute
    # lookups to find the actual function we want to inline
    while isinstance(fn_value, PhiValue) and len(fn_value.values) == 1:
        fn_value = fn_value.values[0]
    if fn_value.parent is not None and fn_value.name is not None:
        parent_value, attr_lookups = find_method_through_phi_parent(fn_value.parent)
        attr_lookups.append(fn_value.name)
        return parent_value, attr_lookups
    return fn_value, []


def inline_method_call(gr, n):  # criterion?
    found_block = False
    for i_bl, bl in enumerate(gr.blocks):
        for i_n, n1 in enumerate(bl.nodes):
            if n1 == n:  # is?
                found_block = True
                break
    assert found_block
    if n.i.opname == "CALL_METHOD":
        fn_parent_value, attr_lookups = find_method_through_phi_parent(n.inputs[0])
        if fn_parent_value.value is None:
            raise NotImplementedError("cannot inline non-explicit function")

        fn_value = fn_parent_value.value
        for al in attr_lookups:
            fn_value = getattr(fn_value, al)

        ## TODO: value for self arg in Method calls?
        ### in general: What is with callables here?
        if isinstance(fn_value, torch.nn.Module):
            mod1 = fn_value
            value_for_self1 = n.inputs[0]
            fn_value = fn_value.forward
        elif inspect.ismethod(fn_value):
            mod1 = fn_value.__self__
            value_for_self1 = n.inputs[1]
        else:
            mod1 = None
            value_for_self1 = None

        if inspect.isbuiltin(fn_value):
            raise NotImplementedError("cannot inline built-in (C-implemented) function")
    else:
        raise NotImplementedError(f"inlining {n}")

    nbl = split_block(gr, bl, bl.nodes[i_n + 1])
    n1 = bl.nodes.pop(i_n)
    assert n1 is n

    gr1 = acquire_method(fn_value, module=mod1, mro_klass=gr.mro_klass if mod1 == gr.module else None)
    make_ssa(gr1)
    make_single_return(gr1)

    # there should be exactly one
    (ret_bl,) = (bl for bl in gr1.blocks if len(bl.nodes) > 0 and bl.nodes[-1].i.opname == "RETURN_VALUE")

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
    gr1.blocks[0].jump_sources = [bl.nodes[-1]]
    ret_node.jump_targets = [((0, 0), nbl)]
    nbl.jump_sources = [ret_node if js == bl.nodes[-1] else js for js in nbl.jump_sources]

    gr.blocks[i_bl + 1 : i_bl + 1] = gr1.blocks

    if gr1.ismethod:
        call_args = [value_for_self1, *n.inputs[2:]]
    else:
        call_args = n.inputs[2:]

    assert len(n.outputs) == 1
    bl.block_outputs.remove(n.outputs[0])  # TODO: what with inplace!!
    bl.block_outputs.update(call_args)
    specify_inputs(gr1, call_args)

    # output values...
    rv = ret_node.inputs.pop()
    assert not ret_node.inputs
    (orv,) = n.outputs
    replace_values(gr, {orv: rv})
    ret_bl.block_outputs.add(rv)


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


def merge_two_blocks(gr, bl1):
    jt = bl1.nodes[-1].jump_targets
    if len(jt) != 1:
        raise RuntimeError("can only fuse blocks with deterministic connection")
    bl2 = jt[0][1]
    if len(bl2.jump_sources) != 1 or bl2.jump_sources[0] != bl1.nodes[-1]:
        raise RuntimeError("second block to be fused must only have first block as jump source")

    replacements = {}
    for i in bl2.block_inputs:
        assert isinstance(i, PhiValue) and len(i.values) == 1
        (iv,) = i.values
        if iv in bl1.block_outputs:
            replacements[i] = iv
        else:
            bl1.block_inputs.append(i)

    replace_values(bl2, replacements)
    # TODO: should this happen automatically in replace_values?

    for o in bl1.block_outputs:
        o.phi_values = [pv for pv in o.phi_values if pv not in replacements]
    bl1.block_outputs = {o for o in bl1.block_outputs if o.phi_values}
    bl1.block_outputs.update(bl2.block_outputs)

    bl1.nodes[-1:] = bl2.nodes
    gr.blocks.remove(bl2)


def merge_blocks_where_possible(gr):
    i_bl = 0
    while i_bl + 1 < len(gr.blocks):
        bl1 = gr.blocks[i_bl]
        jt = bl1.nodes[-1].jump_targets
        if len(jt) == 1:
            bl2 = jt[0][1]
        else:
            bl2 = None
        if bl2 is not None and len(bl2.jump_sources) == 1 and bl2.jump_sources[0] == bl1.nodes[-1]:
            merge_two_blocks(gr, bl1)
        else:
            i_bl += 1


def find_blocks_of_for(for_block):
    assert for_block.nodes[-1].i.opname == "FOR_ITER"

    blocks_of_for_loop = {for_block}
    currently_looking_at = set()

    def find_blocks_of_for_rec(for_block, start_block):
        if for_block == start_block:
            return True
        if start_block in currently_looking_at:
            return False
        currently_looking_at.add(start_block)
        found = False
        for _, jt in start_block.nodes[-1].jump_targets:
            found |= find_blocks_of_for_rec(for_block, jt)
        currently_looking_at.remove(start_block)
        if found:
            blocks_of_for_loop.add(start_block)
        return found

    find_blocks_of_for_rec(for_block, gr.blocks[1].nodes[-1].jump_targets[0][1])
    return blocks_of_for_loop
