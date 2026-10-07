"""Sampled process RSS and CPU time (not system-wide memory attribution)."""
import threading
import time
import psutil


class ResourceMonitor:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.cpu = sum(self.process.cpu_times()[:2])
        def sample():
            while not self.stop.wait(.1):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.seconds = time.perf_counter() - self.started
        self.peak_mb = max(self.peak, self.process.memory_info().rss)/1024**2
        self.cpu_seconds = sum(self.process.cpu_times()[:2]) - self.cpu
