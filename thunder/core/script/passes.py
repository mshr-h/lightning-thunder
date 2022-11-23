import inspect

from .frontend import acquire_method, make_single_return, make_ssa
from .graph import Block


def specify_inputs(gr, inps):
    inp_map = {p: v for p, v in zip(gr.local_variables_at_start, inps)}
    for bl in gr.blocks:
        for n in bl.nodes:
            n.inputs = [inp_map.get(i, i) for i in n.inputs]


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
    nbl.continue_at = bl.continue_at
    bl.continue_at = nbl
    gr.blocks.insert(i, nbl)
    return nbl


def inline_method_calls(gr):  # criterion?
    # node_map = {}
    i_bl = 0
    while i_bl < len(gr.blocks):
        bl = gr.blocks[i_bl]
        new_nodes = []
        i_n = 0
        while i_n < len(bl.nodes):
            n = bl.nodes[i_n]
            if n.i.opname == "CALL_METHOD":
                print("###cm#", n.i.opname, n.inputs[0], n.inputs[1], n.inputs[0].value)
            if (
                n.i.opname == "CALL_METHOD"
                and n.inputs[0].value is not None
                and not inspect.isbuiltin(n.inputs[0].value)
            ):
                print("###inlining", n.outputs)
                nbl = split_block(gr, bl, n)
                assert nbl.nodes[0] == n
                del nbl.nodes[0]
                # print("inline", i, n, n.inputs[0], n.inputs[0].value is None)
                print(n.inputs[1].value)
                gr1 = acquire_method(n.inputs[0].value, module=n.inputs[1].value, mro_klass=gr.mro_klass)
                make_ssa(gr1)
                make_single_return(gr1)
                # there should be exactly one
                (ret_bl,) = (bl for bl in gr1.blocks if len(bl.nodes) == 1 and bl.nodes[0].i.opname == "RETURN_VALUE")

                # print("inline")
                # print_graph(gr1)
                specify_inputs(gr, n.inputs[2:])
                for bl1 in gr1.blocks:
                    if bl1.continue_at == ret_bl:
                        bl1.continue_at = nbl
                    for n1 in bl1.nodes:
                        n1.jump_targets = [nbl if jt == ret_bl else jt for jt in n1.jump_targets]
                # output values...
                (rv,) = ret_bl.nodes[0].inputs
                print("######orv#", n)
                (orv,) = n.outputs
                for bl1 in gr.blocks:
                    for n1 in bl1.nodes:
                        n1.inputs = [inp if inp != orv else rv for inp in n1.inputs]
                gr.blocks[i_bl + 1 : i_bl + 1] = gr1.blocks[:-1]
                # ??? JUMP or what.
            else:
                new_nodes.append(n)
            i_n += 1
        bl.nodes = new_nodes
        i_bl += 1
