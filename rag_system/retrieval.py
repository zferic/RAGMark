import faiss
import numpy as np
import torch


class Retriever:
    def __init__(self, index_path, corpus_texts, gpu_ids=None):
        """
        gpu_ids:
            None      → keep on CPU
            [0]       → single GPU
            [1,2]     → shard across GPUs 1 and 2
        """

        self.texts = corpus_texts
        self.is_gpu_index = False
        self.device = None

        index = faiss.read_index(index_path)
        print("[Retriever] FAISS index loaded on CPU")

        # --------------------------------------------------
        # GPU handling
        # --------------------------------------------------
        if gpu_ids is not None and len(gpu_ids) > 0:

            available = faiss.get_num_gpus()
            if available == 0:
                print("[Retriever] No GPUs available. Staying on CPU.")
            else:
                print(f"[Retriever] Requested GPUs: {gpu_ids}")

                resources = [faiss.StandardGpuResources() for _ in gpu_ids]

                if len(gpu_ids) == 1:
                    # Single GPU
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = True  # reduces memory
                    index = faiss.index_cpu_to_gpu(
                        resources[0],
                        gpu_ids[0],
                        index,
                        co
                    )
                    print(f"[Retriever] Index moved to GPU:{gpu_ids[0]}")

                else:
                    # Multi-GPU sharded
                    co = faiss.GpuMultipleClonerOptions()
                    co.shard = True
                    co.useFloat16 = True

                    index = faiss.index_cpu_to_gpu_multiple_py(
                        [faiss.StandardGpuResources() for _ in gpu_ids],
                        index,
                        co,
                        gpu_ids
                    )

                    print(f"[Retriever] Index sharded across GPUs: {gpu_ids}")

                self.is_gpu_index = True
                self.device = torch.device(f"cuda:{gpu_ids[0]}")

        else:
            print("[Retriever] Staying on CPU")
            print("[Retriever] Index Type:", type(index))

        self.index = index

    @staticmethod
    def to_faiss(x):
        """
        Convert torch tensor (CPU/GPU) or numpy -> contiguous float32 numpy [n, d]
        """
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.is_cuda:
                x = x.cpu()
            x = x.float().numpy()
        x = np.asarray(x, dtype="float32")
        if x.ndim == 1:
            x = x[None, :]
        return np.ascontiguousarray(x, dtype="float32")

    # -------------------------------------------------------------------------
    # Single query
    # -------------------------------------------------------------------------
    def retrieve(self, query_embedding, k=5):
        query_np = self.to_faiss(query_embedding)
        D, I = self.index.search(query_np, k)
        n = len(self.texts)
        return [self.texts[i] for i in I[0] if 0 <= i < n]

    # -------------------------------------------------------------------------
    # Batch queries
    # -------------------------------------------------------------------------
    def batch_retrieve(self, embeddings, k=5):
        emb_np = self.to_faiss(embeddings)
        D, I = self.index.search(emb_np, k)
        n = len(self.texts)
        return [[self.texts[j] for j in row if 0 <= j < n] for row in I]
