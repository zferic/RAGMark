from abc import ABC, abstractmethod


class Pipeline(ABC):
    def __init__(self, embedder, retriever, generator, compressor, args,
                 embed_device, gen_device, index_device, reranker=None):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.compressor = compressor
        self.reranker = reranker
        self.args = args
        self.embed_device = embed_device
        self.gen_device = gen_device
        self.index_device = index_device

    @abstractmethod
    def run_single(self, question):
        """Returns (answer, docs, raw_output, timing)"""
        raise NotImplementedError

    @abstractmethod
    def run_batch(self, questions):
        """Returns (answers, docs_batch, timing)"""
        raise NotImplementedError
