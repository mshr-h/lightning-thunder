# This is a "TorchScript-like" graph representation of Python IR.
# The idea is that blocks are "simple blocks" in terms of the code flow graph,
# i.e. without branches
import copy
import inspect


class NULL:
    """marker for non-existant object."""

    pass


def _make_set(s):
    if isinstance(s, set):
        return s
    return set(s)


class MROAwareObjectRef:  # or as they call it super
    def __init__(self, obj, start_klass=None):
        self.obj = obj
        self.start_klass = start_klass

    def __getattr__(self, name):
        print("###", self.obj, self.start_klass, name)
        ## handle non-methods...
        i = 0
        mro = inspect.getmro(self.obj.value.__class__)
        if self.start_klass is not None:
            while i < len(mro) and not mro[i] == self.start_klass:
                i += 1
            i += 1
        while i < len(mro) and not hasattr(mro[i], name):
            i += 1
        if i >= len(mro):
            raise AttributeError(f"{name} not a member")
        return getattr(mro[i], name)


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
        is_function_arg=False,
    ):
        self.n = n
        self.nr = nr
        self.typ = typ if typ is not None or value is None else type(value)
        self.value = value
        self.name = name
        self.parent = parent
        self.is_global = is_global
        self.is_const = is_const
        self.is_function_arg = is_function_arg
        self.phi_values = []

    def clone(self, translation_dict=None):
        # clones a node, including (recursively) parent nodes
        # uses translation_dict to look up parent node
        # updates translation_dict
        # does not register phi_values on the clone
        # always clone parents?
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            return translation_dict[self]
        parent = self.parent
        if parent:
            if parent in self.translation_dict:
                parent = self.translation_dict[parent]
            else:
                parent = parent.clone(translation_dict=translation_dict)
        v = Value(
            n=self.n,
            nr=self.nr,
            typ=self.typ,
            value=self.value,
            name=self.name,
            parent=parent,
            is_global=self.is_global,
            is_const=self.is_const,
            is_function_arg=self.is_function_arg,
        )
        if translation_dict is not None:
            translation_dict[self] = v
        return v

    def __str__(self):
        parts = []
        if self.is_function_arg:
            parts.append("funcarg")
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
        return f"""{type(self).__name__}({' '.join(parts)})"""

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"


class PhiValue(Value):
    # node?
    def __init__(self, values, jump_sources, block):
        super().__init__()
        self._unfinished_clone = False
        self.block = block
        self._set_values_jump_sourcess(values, jump_sources)

    def _set_values_jump_sourcess(self, values, jump_sources):
        self.values = list(values)
        for v in self.values:
            if v is not None:
                v.phi_values.append(self)
        self.jump_sources = jump_sources

    def clone(self, translation_dict=None):
        # due to loops in the Graph, this is complicated:
        # we do not translate values or jump_sources here, but do
        # translate blocks.
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            return self
        v = PhiValue(self.values, self.jump_sources, translation_dict[self.block])
        v._unfinished_clone = True
        translation_dict[self] = v
        return v

    def post_process_clone(self, *, translation_dict):
        assert self._unfinished_clone
        self._unfinished_clone = False
        self._set_values_jump_sourcess(
            [translation_dict.get(v, v) for v in self.values],
            [translation_dict.get(js, js) for js in self.jump_sources],
        )

    def add_missing_value(self, v, idx):
        assert 0 <= idx < len(self.values)
        assert self.values[idx] == None
        self.values[idx] = v
        v.phi_values.append(self)


def unify_values(values, jump_sources, bl, all_predecessors_done=True):
    if all_predecessors_done:
        if len(values) == 1:
            return values[0]
        val = values[0]
        if all(v is val for v in values[1:]):
            return val
        # different values
    return PhiValue(values, jump_sources, bl)


# A node corresponds to one Python bytecode instruction
class Node:
    def __init__(self, *, i=None, inputs=None, outputs=None, line_no=None):
        self.i = i
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if inputs is not None else []
        self.jump_targets = []
        self.line_no = line_no
        self.block = None

    def clone(self, translation_dict=None):
        """.block of the clone will be None if block is not in translation dict"""
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            return translation_dict[self]
        inputs = [i.clone(translation_dict=translation_dict) for i in self.inputs]
        outputs = [o.clone(translation_dict=translation_dict) for o in self.outputs]
        i = copy.copy(self.i)
        n2 = Node(i=i, inputs=inputs, outputs=outputs, line_no=self.line_no)
        n2.jump_targets = [(se, translation_dict.get(bl, bl)) for se, bl in self.jump_targets]
        n2.block = translation_dict.get(self.block)
        translation_dict[self] = n2
        return n2

    def __str__(self):
        # i.i.offset // 2, i.i.opname, i.i.arg, "(", i.i.argval, ")"
        if self.i.opname == "CALL_METHOD":
            return f"CALL_METHOD({self.inputs})"
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
    def __init__(self, is_ssa=True):
        self.is_ssa = is_ssa
        self.jump_sources = []
        self.nodes = []

    def __str__(self):
        return "\n".join([f"  Block (reached from {self.jump_sources})"] + ["    " + str(n) for n in self.nodes])

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"

    def insert_node(self, n, insert_after=None, insert_before=None):
        assert n.block is None
        if insert_after is None and insert_before is None:
            if self.is_ssa:
                raise ValueError("need to supply insert_after or insert_before")
            else:
                self.nodes.append(n)
                # validity checks? (also below)
                n.block = self
                return
        elif insert_after is not None and insert_before is not None:
            raise ValueError("only one of insert_after or insert_before can be supplied")
            # this is the usual case.
            # in the pre-ssa graph, both None mean to insert at the end.
            assert insert_after is not None or insert_before is not None

        to_find = insert_after or insert_before
        for idx, n2 in enumerate(self.nodes):
            if n2 is to_find:
                break
        if n2 is not to_find:
            raise ValueError(f"could not find node {n}")

        # validity checks? (also above)
        n.block = self
        if insert_after:
            self.nodes.insert(idx + 1, n)
        else:
            self.nodes.insert(idx, n)


class Graph:
    def __init__(self, blocks=None):
        self.blocks = [] if blocks is None else blocks

    def __str__(self):
        return "\n".join(["Graph of"] + [str(b) for b in self.blocks])

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"

    def nodes(self):
        for b in self.blocks:
            for n in self.nodes:
                yield n

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


def replace_values(gr_or_bl, value_map, follow_phi_values=False):
    ### Replacing a value:
    # - as inputs/outputs of nodes
    # - value.parent for other values
    # - phi nodes
    # - graph input (?) / initial vars

    def map_values(v):
        if v in value_map:
            if follow_phi_values:
                for pv in v.phi_values:
                    pv.values = [(vv if vv is not v else value_map[v]) for vv in pv.values]
                value_map[v].phi_values += v.phi_values
                v.phi_values = []
            return value_map[v]
        if isinstance(v.value, MROAwareObjectRef):
            v.value.obj = map_values(v.value.obj)
        if v.parent is not None:
            v.parent = map_values(v.parent)
        if isinstance(v, PhiValue):
            new_values = [map_values(vv) for vv in v.values]
            for ov, nv in zip(v.values, new_values):
                ov.phi_values.remove(v)
                nv.phi_values.append(v)
            v.values = new_values
        return v

    def process_block(bl):
        bl.block_inputs = [map_values(vv) for vv in bl.block_inputs]
        for n in bl.nodes:
            n.inputs = [map_values(vv) for vv in n.inputs]
            n.outputs = [map_values(vv) for vv in n.outputs]

    if isinstance(gr_or_bl, Graph):
        for bl in gr_or_bl.blocks:
            process_block(bl)
    elif isinstance(gr_or_bl, Block):
        process_block(gr_or_bl)
    else:
        raise TypeError("replace_values works on Graph or Block objects")


## TODO: our should this be a method?
def make_dot(gr, format="png", add_names=False):
    import graphviz

    dot = graphviz.Digraph(name="thunder_graph", format=format)

    block_idxes = {}

    value_idxes = {}

    for i_bl, bl in enumerate(gr.blocks):
        block_idxes[bl] = i_bl
        with dot.subgraph(name=f"cluster_bl_{i_bl}") as sub_dot:
            for i_i, i in enumerate(bl.block_inputs):
                i_nr = len(value_idxes)
                value_idxes[i] = i_nr
                i_name = f"bi %{i_nr}"
                if add_names:
                    i.name = i_name
                v_color = "black" if i not in bl.block_outputs else "red"
                sub_dot.node(f"v {i_nr}", label=i_name, color=v_color)

            for i_n, n in enumerate(bl.nodes):
                label = n.i.opname
                if n.i.opname == "CALL_METHOD":
                    label = "CM " + n.inputs[0].name
                sub_dot.node(f"i {i_bl} {i_n}", label, shape="box")
                for o in n.outputs:
                    if o not in value_idxes:
                        o_nr = len(value_idxes)
                        value_idxes[o] = o_nr
                        o_name = o.name or f"%{o_nr}"
                        if add_names:
                            o.name = o_name
                        v_color = "black" if o not in bl.block_outputs else "red"
                        sub_dot.node(f"v {o_nr}", label=o_name, color=v_color)
                    else:
                        o_nr = value_idxes[o]
                    sub_dot.edge(f"i {i_bl} {i_n}", f"v {o_nr}", color="blue")
                if i_n > 0:
                    sub_dot.edge(f"i {i_bl} {i_n - 1}", f"i {i_bl} {i_n}")

    for i_bl, bl in enumerate(gr.blocks):
        for _, jt_bl in bl.nodes[-1].jump_targets:
            dot.edge(f"i {i_bl} {len(bl.nodes) - 1}", f"i {block_idxes[jt_bl]} {0}")
        for i in bl.block_inputs:
            i_idx = value_idxes[i]
            if isinstance(i, PhiValue):
                for v in i.values:
                    if v in value_idxes:
                        dot.edge(f"v {value_idxes[v]}", f"v {i_idx}", color="green")

        for i_n, n in enumerate(bl.nodes):
            for i in n.inputs:
                if i in value_idxes:
                    dot.edge(f"v {value_idxes[i]}", f"i {i_bl} {i_n}", color="blue")
                elif isinstance(i, PhiValue):
                    print("oops", repr(i))
                    for v in i.values:
                        if v in value_idxes:
                            dot.edge(f"v {value_idxes[v]}", f"i {i_bl} {i_n}", color="red")

    return dot


def clone_blocks(blocks_to_clone: list[Block], translation_dict=None):
    assert all(bl.is_ssa for bl in blocks_to_clone)
    if translation_dict is None:
        translation_dict = {}

    blocks_todo = []
    for obl in blocks_to_clone:
        if obl not in translation_dict:
            bl = Block()
            translation_dict[obl] = bl
            blocks_todo.append(obl)

    for obl in blocks_todo:
        bl = translation_dict[obl]
        bl.block_inputs = [i.clone(translation_dict=translation_dict) for i in obl.block_inputs]
        bl.block_outputs = {o.clone(translation_dict=translation_dict) for o in obl.block_outputs}
        bl.nodes = [n.clone(translation_dict=translation_dict) for n in obl.nodes]
    for obl in blocks_todo:
        bl = translation_dict[obl]
        bl.jump_sources = [translation_dict.get(js, js) for js in obl.jump_sources]
        for i in bl.block_inputs:
            i.post_process_clone(translation_dict=translation_dict)
    return [translation_dict[bl] for bl in blocks_to_clone], translation_dict
