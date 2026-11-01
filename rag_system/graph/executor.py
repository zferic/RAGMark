"""Runs a GraphSpec: topological node order, optional feedback loop, timing epilogue."""
from graphlib import TopologicalSorter

from ..timing import RAGTiming
from .nodes import CONDITIONS, NODE_TYPES, StageContext
from .spec import GraphSpec

_TOP_LEVEL_KEYS = ("ttft", "decode_time", "generate_total_time", "tokens_generated",
                   "tokens_per_second", "itl_mean", "itl_p50", "itl_p95", "itl_max",
                   "context_tokens_before", "context_tokens_after")


class GraphExecutor:
    def __init__(self, spec: GraphSpec, components: dict, args, devices: dict):
        self.spec = spec
        self.components = components
        self.args = args
        self.devices = devices
        self.nodes = self._build_nodes()

    def _build_nodes(self):
        ids = [n.id for n in self.spec.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate node ids in graph '{self.spec.name}': {ids}")
        by_id = {}
        for ns in self.spec.nodes:
            if ns.type not in NODE_TYPES:
                raise ValueError(f"node '{ns.id}': unknown type '{ns.type}' "
                                 f"(registered: {sorted(NODE_TYPES)})")
            node = NODE_TYPES[ns.type](ns.id, ns.params, ns.component)
            if node.component_name and node.component_name not in self.components:
                raise ValueError(f"node '{ns.id}' ({ns.type}) needs component "
                                 f"'{node.component_name}' which was not provided")
            for dep in ns.after:
                if dep not in ids:
                    raise ValueError(f"node '{ns.id}' runs after unknown node '{dep}'")
            by_id[ns.id] = node
        until = self.spec.loop.until
        if until and until not in CONDITIONS:
            raise ValueError(f"unknown loop condition '{until}' (registered: {sorted(CONDITIONS)})")
        order = TopologicalSorter({n.id: n.after for n in self.spec.nodes}).static_order()
        return [by_id[i] for i in order]

    def run(self, questions, batch, tracer=None):
        """Returns (state, timing). state['answers'] / state['docs'] hold the outputs."""
        loop = self.spec.loop
        timing = RAGTiming()
        timing.start("rag_total")
        ctx = StageContext(self.components, self.args, self.devices, timing, tracer, batch)
        state = {"questions": list(questions), "queries": list(questions)}

        last = 0
        for i in range(loop.iterations):
            last = i
            ctx.set_iter(i, loop.iterations, self.spec.timing_suffix)
            for node in self.nodes:
                node.run(state, ctx)
            if i == loop.iterations - 1:
                break
            if loop.until and CONDITIONS[loop.until](state, ctx):
                break
            state["queries"] = [loop.feedback.format(question=q, answer=a)
                                for q, a in zip(state["questions"], state["answers"])]
        timing.end("rag_total")
        self._finalize_timing(timing, last)
        return state, timing

    def _finalize_timing(self, timing, last_iter):
        data = timing.data
        if self.spec.timing_suffix == "iter":
            # top-level keys mirroring the final round, for CSV compatibility with standard runs
            for key in _TOP_LEVEL_KEYS:
                src = f"{key}_iter_{last_iter}"
                if src in data:
                    data[key] = data[src]
        else:
            # standard schema also exposes everything under *_iter_0
            for key in list(data.keys()):
                if not key.endswith("_iter_0"):
                    data[f"{key}_iter_0"] = data[key]
