from ..graph import GraphExecutor, GraphSpec, default_spec
from .base import Pipeline


class GraphPipeline(Pipeline):
    """
    Pipeline backed by a declarative graph (rag_system/graph).

    Subclasses set `kind` to pick a built-in graph; the generic `graph` pipeline
    loads one from the YAML file given as args.graph_spec.
    """
    kind = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.spec = self.build_spec()
        components = {
            "embedder": self.embedder,
            "retriever": self.retriever,
            "generator": self.generator,
            "compressor": self.compressor,
            "reranker": self.reranker,
        }
        self.executor = GraphExecutor(
            self.spec,
            {k: v for k, v in components.items() if v is not None},
            self.args,
            {"embed": self.embed_device, "gen": self.gen_device, "index": self.index_device},
        )

    def build_spec(self):
        graph_spec = getattr(self.args, "graph_spec", None)
        if graph_spec:
            return GraphSpec.from_yaml(graph_spec)
        return default_spec(
            self.kind,
            use_rag=self.args.use_rag,
            has_reranker=self.reranker is not None,
            has_compressor=self.compressor is not None,
            iterations=getattr(self.args, "iter_num", 2),
        )

    def run_single(self, question, tracer=None):
        state, timing = self.executor.run([question], batch=False, tracer=tracer)
        return state["answers"][0], state.get("docs", ""), state["raw"], timing

    def run_batch(self, questions, tracer=None):
        state, timing = self.executor.run(questions, batch=True, tracer=tracer)
        return state["answers"], state.get("docs", []), timing


class StandardGraphPipeline(GraphPipeline):
    kind = "standard"


class IterativeGraphPipeline(GraphPipeline):
    kind = "iterative"
