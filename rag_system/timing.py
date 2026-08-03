# rag_system/timing.py
import time
import torch
import json
import os
import threading
from contextlib import contextmanager
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import pynvml


_proc = psutil.Process(os.getpid())



# -----------------------------------------------------------------------------
# High-level timer
# -----------------------------------------------------------------------------
class Timer:
    """Simple wall-clock timer for measuring elapsed time."""
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time


# -----------------------------------------------------------------------------
# GPU Memory Helpers
# -----------------------------------------------------------------------------
def get_gpu_memory(device=None):
    if not torch.cuda.is_available():
        return {"allocated": None, "peak": None}

    if device is not None:
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device) / 1e6
        peak = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e6

    return {"allocated": allocated, "peak": peak}


def reset_gpu_peak(device=None):
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.reset_peak_memory_stats(device)
        else:
            torch.cuda.reset_peak_memory_stats()


def get_nvml_memory(gpu_id: int):
    """Query true GPU memory via pynvml (sees FAISS, not just PyTorch tensors)."""
    try:
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "used_mb": info.used / 1024**2,
            "free_mb": info.free / 1024**2,
        }
    except Exception as e:
        print(f"[pynvml] get_nvml_memory(gpu_id={gpu_id}) failed: {e}")
        return {"used_mb": None, "free_mb": None}


# -----------------------------------------------------------------------------
# Power Poller
# -----------------------------------------------------------------------------
class PowerPoller:
    """
    Background thread that samples GPU power via pynvml at a fixed interval.
    Use to estimate energy consumption (joules) for a block of work.

    Usage:
        poller = PowerPoller(interval=0.05)
        poller.start()
        # ... do work ...
        poller.stop()
        energy  = poller.energy_joules()   # {gpu_id: joules}
        power   = poller.mean_power_w()    # {gpu_id: watts}
    """

    def __init__(self, gpu_ids=None, interval: float = 0.05):
        self.interval    = interval
        self.gpu_ids     = gpu_ids or list(range(torch.cuda.device_count()))
        self.samples     = {i: [] for i in self.gpu_ids}   # power (W)
        self.util_gpu    = {i: [] for i in self.gpu_ids}   # SM utilization (%)
        self.util_mem    = {i: [] for i in self.gpu_ids}   # memory BW utilization (%)
        self._stop       = threading.Event()
        self._thread     = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        try:
            pynvml.nvmlInit()
            handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_ids}
        except Exception as e:
            print(f"[PowerPoller] nvml init failed: {e}")
            return
        while not self._stop.is_set():
            for i, handle in handles.items():
                try:
                    mw   = pynvml.nvmlDeviceGetPowerUsage(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.samples[i].append(mw / 1000.0)
                    self.util_gpu[i].append(util.gpu)
                    self.util_mem[i].append(util.memory)
                except Exception:
                    self.samples[i].append(None)
                    self.util_gpu[i].append(None)
                    self.util_mem[i].append(None)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def energy_joules(self):
        """Estimated energy per GPU in joules (∫ P dt ≈ sum(samples) × interval)."""
        return {
            i: sum(s for s in slist if s is not None) * self.interval
            for i, slist in self.samples.items()
        }

    def mean_power_w(self):
        """Mean power draw per GPU in watts."""
        result = {}
        for i, slist in self.samples.items():
            valid = [s for s in slist if s is not None]
            result[i] = sum(valid) / len(valid) if valid else None
        return result

    def mean_utilization(self):
        """Mean SM and memory BW utilization per GPU (%)."""
        result = {}
        for i in self.gpu_ids:
            gpu_valid = [s for s in self.util_gpu[i] if s is not None]
            mem_valid = [s for s in self.util_mem[i] if s is not None]
            result[i] = {
                "sm_pct":  sum(gpu_valid) / len(gpu_valid) if gpu_valid else None,
                "mem_pct": sum(mem_valid) / len(mem_valid) if mem_valid else None,
            }
        return result

    def peak_utilization(self):
        """Peak SM and memory BW utilization per GPU (%)."""
        result = {}
        for i in self.gpu_ids:
            gpu_valid = [s for s in self.util_gpu[i] if s is not None]
            mem_valid = [s for s in self.util_mem[i] if s is not None]
            result[i] = {
                "sm_pct":  max(gpu_valid) if gpu_valid else None,
                "mem_pct": max(mem_valid) if mem_valid else None,
            }
        return result


# -----------------------------------------------------------------------------
# Trace Poller — continuous timeseries + stage markers for visualization
# -----------------------------------------------------------------------------
class TracePoller:
    """
    Continuous GPU timeseries recorder with stage marker support.
    Samples power, SM utilization, memory BW utilization, and NVML memory
    at a fixed interval. Stage boundaries are recorded via mark() and stored
    as timestamps relative to poller start.

    Use for a small number of representative runs to generate figures
    like resource utilization breakdowns (power/util vs time with stage shading).

    Usage:
        poller = TracePoller(interval=0.05)
        poller.start()
        poller.mark("embed_start")
        embedder.embed(q)
        poller.mark("embed_end")
        poller.mark("faiss_start")
        retriever.retrieve(...)
        poller.mark("faiss_end")
        poller.mark("generate_start")
        generator.generate(...)
        poller.mark("generate_end")
        poller.stop()
        trace = poller.get_trace()   # dict with "samples" and "markers"
        poller.save(path)            # write to jsonl
    """

    def __init__(self, gpu_ids=None, interval: float = 0.05):
        self.interval   = interval
        self.gpu_ids    = gpu_ids or list(range(torch.cuda.device_count()))
        self.markers    = {}          # label -> elapsed seconds
        self._samples   = []          # list of dicts, one per tick
        self._stop      = threading.Event()
        self._thread    = None
        self._t0        = None

    def start(self):
        self._stop.clear()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def mark(self, label: str):
        """Record a stage boundary timestamp."""
        self.markers[label] = time.time() - self._t0

    def _poll(self):
        try:
            pynvml.nvmlInit()
            handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_ids}
        except Exception as e:
            print(f"[TracePoller] nvml init failed: {e}")
            return
        while not self._stop.is_set():
            t = time.time() - self._t0
            row = {"t": t}
            for i, handle in handles.items():
                try:
                    mw   = pynvml.nvmlDeviceGetPowerUsage(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    row[f"gpu{i}_power_w"]  = mw / 1000.0
                    row[f"gpu{i}_sm_pct"]   = util.gpu
                    row[f"gpu{i}_mem_pct"]  = util.memory
                    row[f"gpu{i}_mem_mb"]   = mem.used / 1024**2
                except Exception:
                    row[f"gpu{i}_power_w"]  = None
                    row[f"gpu{i}_sm_pct"]   = None
                    row[f"gpu{i}_mem_pct"]  = None
                    row[f"gpu{i}_mem_mb"]   = None
            # CPU utilization (process)
            row["cpu_pct"] = _proc.cpu_percent()
            self._samples.append(row)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def compute_stage_durations(self) -> dict:
        """Sum active time per stage across all iterations from markers."""
        import re
        stage_names = ["embed", "faiss", "rerank", "compress", "generate"]
        durations = {}
        iter_keys = [k for k in self.markers if re.search(r'_iter_\d+$', k)]
        if iter_keys:
            n_iters = max(int(re.search(r'_iter_(\d+)$', k).group(1)) for k in iter_keys) + 1
            for stage in stage_names:
                total = sum(
                    self.markers[f"{stage}_end_iter_{i}"] - self.markers[f"{stage}_start_iter_{i}"]
                    for i in range(n_iters)
                    if f"{stage}_start_iter_{i}" in self.markers and f"{stage}_end_iter_{i}" in self.markers
                )
                if total > 0:
                    durations[stage] = total
        else:
            for stage in stage_names:
                s = self.markers.get(f"{stage}_start")
                e = self.markers.get(f"{stage}_end")
                if s is not None and e is not None:
                    durations[stage] = e - s
        return durations

    def get_trace(self):
        return {"markers": self.markers, "samples": self._samples}

    def save(self, path: str, meta: dict = None):
        """Save full trace to a jsonl file (one line = full trace for this query)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = {
            "markers": self.markers,
            "samples": self._samples,
            "trace_stage_durations": self.compute_stage_durations(),
        }
        if meta:
            record["meta"] = meta
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


# -----------------------------------------------------------------------------
# PyTorch Profiler Wrapper
# -----------------------------------------------------------------------------
class TorchProfiler:
    """
    Wrapper for PyTorch profiler with:
        - CPU & CUDA profiling
        - Chrome Trace output
        - Optional wrapping of embed + decode operations
    """
    def __init__(self, save_path=None):
        self.save_path = save_path
        self.prof = None

    def __enter__(self):
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.prof = profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            use_cuda=torch.cuda.is_available()
        )
        self.prof.__enter__()
        return self.prof

    def __exit__(self, exc_type, exc_value, traceback):
        self.prof.__exit__(exc_type, exc_value, traceback)

        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.prof.export_chrome_trace(self.save_path)

        # Return profiler object for summary extraction
        return False


# -----------------------------------------------------------------------------
# High-Level Timing for Each Stage of RAG
# -----------------------------------------------------------------------------
class RAGTiming:
    """
    Tracks:
        - Stage wall-clock timing
        - CPU RSS memory
        - GPU memory per device
        - Token counts
    """

    def __init__(self):
        self.data = {}

    # -----------------------------
    # Wall-clock timing
    # -----------------------------
    def start(self, key: str):
        self.data[key + "_timer"] = time.time()

    def end(self, key: str):
        start = self.data.get(key + "_timer")
        self.data[key] = 0 if start is None else time.time() - start
    # -----------------------------
    # CPU memory
    # -----------------------------
    def capture_cpu_memory(self, label):
        self.data[label] = _proc.memory_info().rss / 1024**2

    # -----------------------------
    # GPU memory (device-aware)
    # -----------------------------
    def capture_gpu_memory(self, label, device=None):
        """
        Capture GPU memory for a specific device.

        Args:
            label: name to store in timing data
            device: torch.device or int (e.g. cuda:0)
        """
        if isinstance(device, torch.device):
            device = device.index

        self.data[label] = get_gpu_memory(device)

    def reset_gpu_peak(self, device=None):
        """
        Reset GPU peak memory stats for a specific device.
        """
        if isinstance(device, torch.device):
            device = device.index

        reset_gpu_peak(device)

    # -----------------------------
    # Token counting
    # -----------------------------
    def record_tokens(self, label, count):
        self.data[label] = count

    # -----------------------------
    # Optional: capture all GPUs
    # -----------------------------
    def capture_all_gpu_memory(self, label_prefix):
        """
        Capture memory for all visible CUDA devices.
        """
        if not torch.cuda.is_available():
            return

        for i in range(torch.cuda.device_count()):
            self.data[f"{label_prefix}_gpu_{i}"] = get_gpu_memory(i)

    # -----------------------------
    # NVML memory (sees FAISS + PyTorch)
    # -----------------------------
    def capture_nvml_memory(self, label, gpu_id: int):
        """
        Capture true GPU memory via pynvml for a specific GPU.
        Sees all consumers (FAISS, PyTorch, CUDA context), unlike torch.cuda.memory_allocated.

        Args:
            label: key to store in timing data
            gpu_id: integer GPU index (e.g. 0, 1, 2)
        """
        self.data[label] = get_nvml_memory(gpu_id)

    def capture_all_nvml_memory(self, label_prefix):
        """
        Capture true GPU memory via pynvml for all visible GPUs.
        """
        if not torch.cuda.is_available():
            return

        for i in range(torch.cuda.device_count()):
            self.data[f"{label_prefix}_gpu_{i}"] = get_nvml_memory(i)
    # -----------------------------
    # Generic metric recording
    # -----------------------------
    def record(self, label, value):
        """
        Record any scalar metric (float, int, str, etc.)
        """
        self.data[label] = value

    # -----------------------------
    # Stage markers (for TracePoller)
    # -----------------------------
    def mark(self, label: str, t0: float):
        """Record a stage boundary relative to a pipeline start time t0."""
        self.data[f"mark_{label}"] = time.time() - t0
    # -----------------------------
    # Persistence
    # -----------------------------
    def save_jsonl(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.data) + "\n")

    def save_csv(self, path, is_first_entry=False):
        import csv

        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = is_first_entry or not os.path.exists(path)

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(self.data)


# -----------------------------------------------------------------------------
# Context managers for profiling individual components
# -----------------------------------------------------------------------------
@contextmanager
def profile_block(name, profiler_enabled, trace_path=None):
    """
    Wrap a block with PyTorch profiler if enabled.
    This ensures consistent behavior across benchmark entrypoints.
    """
    if profiler_enabled:
        with TorchProfiler(save_path=trace_path) as prof:
            with record_function(name):
                yield prof
    else:
        with record_function(name):
            yield None
