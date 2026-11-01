"""Declarative graph description: from a dict, a YAML file, or built-in defaults."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NodeSpec:
    id: str
    type: str
    after: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    component: Optional[str] = None


@dataclass
class LoopSpec:
    iterations: int = 1
    # Template that builds the next round's query. Fields: {question}, {answer}.
    feedback: str = "{question} {answer}"
    # Name of a registered condition; the loop stops early when it returns True.
    until: Optional[str] = None


@dataclass
class GraphSpec:
    name: str
    nodes: list
    loop: LoopSpec = field(default_factory=LoopSpec)
    # "none": timing keys have no suffix (aliased to *_iter_0 at the end, standard schema)
    # "iter": every key gets _iter_{i} (iterative schema)
    timing_suffix: str = "none"

    @classmethod
    def from_dict(cls, d):
        nodes = [NodeSpec(id=n["id"], type=n["type"],
                          after=_as_list(n.get("after")),
                          params=n.get("params") or {},
                          component=n.get("component"))
                 for n in d["nodes"]]
        loop = LoopSpec(**(d.get("loop") or {}))
        return cls(name=d.get("name", "graph"), nodes=nodes, loop=loop,
                   timing_suffix=d.get("timing_suffix",
                                       "iter" if loop.iterations > 1 else "none"))

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        d.setdefault("name", path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        return cls.from_dict(d)

    def to_dict(self):
        return {
            "name": self.name,
            "timing_suffix": self.timing_suffix,
            "loop": vars(self.loop),
            "nodes": [vars(n) for n in self.nodes],
        }


def _as_list(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _chain(name, steps, loop, timing_suffix):
    nodes, prev = [], None
    for node_id, type_name in steps:
        nodes.append(NodeSpec(node_id, type_name, after=[prev] if prev else []))
        prev = node_id
    return GraphSpec(name, nodes, loop, timing_suffix)


def default_spec(kind, *, use_rag=True, has_reranker=False, has_compressor=False, iterations=2):
    """
    Graphs equivalent to the former hardcoded StandardPipeline / IterativePipeline.

    standard : [embed -> retrieve -> rerank? -> context -> compress?] -> generate
               (generate only when use_rag is False)
    iterative: same retrieval chain, repeated `iterations` times; each round's
               answer is appended to the query for the next round.
    """
    steps = []
    if use_rag or kind == "iterative":
        steps += [("embed", "embed"), ("retrieve", "retrieve")]
        if has_reranker:
            steps.append(("rerank", "rerank"))
        steps.append(("context", "context"))
        if has_compressor:
            steps.append(("compress", "compress"))
    steps.append(("generate", "generate"))

    if kind == "standard":
        return _chain("standard", steps, LoopSpec(iterations=1), "none")
    if kind == "iterative":
        spec = _chain("iterative", steps, LoopSpec(iterations=iterations), "iter")
        spec.nodes[-1].params["prompt_type"] = "auto"
        return spec
    raise ValueError(f"unknown default graph kind: {kind}")
