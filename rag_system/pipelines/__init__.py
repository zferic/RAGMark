from .standard import StandardPipeline
from .iterative import IterativePipeline

PIPELINES = {
    "standard": StandardPipeline,
    "iterative": IterativePipeline,
}
