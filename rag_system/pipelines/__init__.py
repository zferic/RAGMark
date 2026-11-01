from .standard import StandardPipeline
from .iterative import IterativePipeline
from .graph import GraphPipeline, StandardGraphPipeline, IterativeGraphPipeline

PIPELINES = {
    "standard": StandardGraphPipeline,
    "iterative": IterativeGraphPipeline,
    # user-defined graph: pass --graph_spec path/to/graph.yaml
    "graph": GraphPipeline,
    # original hardcoded implementations, kept for equivalence checks
    "standard_legacy": StandardPipeline,
    "iterative_legacy": IterativePipeline,
}
