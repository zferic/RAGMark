"""
Node interface, registries, and the per-run context handed to every node.

A node reads and writes a shared `state` dict. Timing keys, tracer marks and
GPU/NVML captures are emitted by the node through `StageContext`, so the
column names in the perf CSVs stay identical to the old hardcoded pipelines.
"""
from contextlib import contextmanager

NODE_TYPES = {}
CONDITIONS = {}


def register_node(type_name):
    def deco(cls):
        NODE_TYPES[type_name] = cls
        cls.type_name = type_name
        return cls
    return deco


def register_condition(name):
    """Register a loop-stop predicate: fn(state, ctx) -> True to stop iterating."""
    def deco(fn):
        CONDITIONS[name] = fn
        return fn
    return deco


class Node:
    """
    Subclass and decorate with @register_node("my_type").

    component: default key looked up in the components dict ("embedder",
               "retriever", ...). A node spec can override it with
               `component: other_name`, which lets a graph hold e.g. two generators.
    """
    type_name = None
    component = None

    def __init__(self, node_id, params=None, component=None):
        self.id = node_id
        self.params = dict(params or {})
        self.component_name = component or self.component

    def run(self, state, ctx):
        raise NotImplementedError


class StageContext:
    """Per-run context: components, devices, timing, tracer, iteration suffix."""

    def __init__(self, components, args, devices, timing, tracer, batch):
        self.components = components
        self.args = args
        self.devices = devices      # dict: embed, gen, index
        self.timing = timing
        self.tracer = tracer
        self.batch = batch
        self.iter_idx = 0
        self.num_iters = 1
        self.sfx = ""
        self.suffix_mode = "none"   # "none" or "iter"

    def set_iter(self, i, n, suffix_mode):
        self.iter_idx, self.num_iters, self.suffix_mode = i, n, suffix_mode
        self.sfx = f"_iter_{i}" if suffix_mode == "iter" else ""

    @property
    def last_iter(self):
        return self.iter_idx == self.num_iters - 1

    def comp(self, node):
        try:
            return self.components[node.component_name]
        except KeyError:
            raise KeyError(f"node '{node.id}' needs component '{node.component_name}', "
                           f"available: {sorted(self.components)}")

    def mark(self, label):
        if self.tracer:
            self.tracer.mark(label + self.sfx)

    @contextmanager
    def stage(self, key):
        """Time a stage and emit {key}_start/{key}_end tracer marks."""
        self.mark(f"{key}_start")
        self.timing.start(key + self.sfx)
        yield
        self.timing.end(key + self.sfx)
        self.mark(f"{key}_end")
