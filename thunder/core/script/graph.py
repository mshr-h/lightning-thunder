# This is a "TorchScript-like" graph representation of Python IR.
# The idea is that blocks are "simple blocks" in terms of the code flow graph,
# i.e. without branches
class Value:
    def __init__(
        self,
        *,
        n=None,
        nr=None,
        typ=None,
        value=None,
        name=None,
        parent=None,
        is_global=False,
        is_const=False,
    ):
        self.n = n
        self.nr = nr
        self.typ = typ if typ is not None or value is None else type(value)
        self.value = value
        self.name = name
        self.parent = parent
        self.is_global = is_global
        self.is_const = is_const
        self.phi_values = []

    def __str__(self):
        parts = []
        if self.name:
            parts.append(f"name={self.name}")
        if self.typ is not None:
            parts.append(f"typ={self.typ}")
        if self.value:
            parts.append(f"value of type {type(self.value)}")
        if self.is_const:
            parts.append("const")
        if self.is_global:
            parts.append("global")
        if self.parent is not None:
            parts.append(f"parent={self.parent}")
        return f"""Value({' '.join(parts)})"""

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"


class UnionValue(Value):
    # node?
    def __init__(self, values, jump_sources, bl):
        super().__init__()
        self.values = values
        for v in self.values:
            v.phi_values.append(self)
        self.bl = bl
        self.jump_sources = jump_sources


def unify_values(values, jump_sources, bl):
    # what to do with loops, really?
    if bl in jump_sources:
        print("ohoh, loop")
    if len(values) == 1:
        return values[0]
    val = values[0]
    if all(v is val for v in values[1:]):
        return val
    # different values
    return UnionValue(values, jump_sources, bl)
    raise Exception(f"unimplemnted {values}")


# A node corresponds to one Python bytecode instruction
class Node:
    def __init__(self, *, i=None, inputs=None, outputs=None, line_no=None):
        self.i = i
        self.inputs = inputs
        self.outputs = outputs
        self.jump_targets = []
        self.line_no = line_no

    def __str__(self):
        # i.i.offset // 2, i.i.opname, i.i.arg, "(", i.i.argval, ")"
        return f"{self.i.opname} {self.i.arg} ({self.i.argval})"  # str(self.i)

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"


# Blocks have the first instruction (only) as the jump target
# (or the function entry point)
# Blocks always have a single final instruction that jumps (or RETURN)
# conditional jumps (including e.g. FOR_ITER) always have the non-jumping
# target first and then the jumping target.
# The jump targets are other blocks and are atributes of the jump instruction.
class Block:
    def __init__(self, is_ssa=False):
        # offset_start=0, stack_at_start=None, i=None, jump_source=None
        self.is_ssa = is_ssa
        # if not is_ssa:
        #    assert stack_at_start is not None
        #    self.stack_at_start = stack_at_start
        #    self.all_stacks_at_start = [(jump_source, self.stack_at_start)]
        #    self.all_local_variables_at_start = []
        self.jump_sources = []
        self.nodes = []  # if i is None else i
        # self.offset_start = offset_start

    def __str__(self):
        return "\n".join([f"  Block (reached from {self.jump_sources})"] + ["    " + str(n) for n in self.nodes])

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"


class Graph:
    def __init__(self, blocks=None):
        self.blocks = [] if blocks is None else blocks

    def __str__(self):
        return "\n".join(["Graph of"] + [str(b) for b in self.blocks])

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"

    def nodes(self):
        for b in self.blocks:
            pass

    def print(self):
        value_counter = 1
        print(self.local_variables_at_start)
        for bl in self.blocks:
            for n in bl.i:
                for o in n.outputs:
                    o.print_name = f"{o.name}:{value_counter}" if o.name is not None else f":{value_counter}"
                    value_counter += 1
                for i in n.inputs:
                    if not hasattr(i, "print_name"):
                        i.print_name = f"{i.name}:{value_counter}" if i.name is not None else f":{value_counter}"
                        value_counter += 1
                av = f"[{n.i.argval}]" if n.i.argval is not None else ""
                print(
                    ",".join(o.print_name for o in n.outputs),
                    "=",
                    n.i.opname,
                    f"{av}(",
                    ", ".join([i.print_name for i in n.inputs]) + ")",
                )


def insert_before(new_n, n):
    idx = n.block.nodes.index(n)
    n.block.nodes.insert(idx, new_n)
    new_n.block = n.block


def insert_after(new_n, n):
    idx = n.block.nodes.index(n)
    n.block.nodes.insert(idx + 1, new_n)
    new_n.block = n.block


def replace_values(gr, value_map):
    ### Replacing a value:
    # - as inputs/outputs of nodes
    # - value.parent for other values
    # - phi nodes
    # - graph input (?) / initial vars

    def map_values(v):
        if v in value_map:
            return value_map[v]
        if v.parent is not None:
            v.parent = map_values(v.parent)
        if isinstance(v, UnionValue):
            # print("###processing union value", v)
            new_values = [map_values(vv) for vv in v.values]
            for ov, nv in zip(v.values, new_values):
                ov.phi_values.remove(v)
                nv.phi_values.append(v)
            v.values = new_values
        return v

    for bl in gr.blocks:
        for n in bl.nodes:
            n.inputs = [map_values(vv) for vv in n.inputs]
            n.outputs = [map_values(vv) for vv in n.outputs]
