import os
import time
import openai
import pytest
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@pytest.fixture(scope="session")
def openai_client():
    if not openai.api_key:
        pytest.skip("OPENAI_API_KEY not set in .env or environment")
    return openai


@pytest.fixture
def measure_latency():
    """Measure latency and assert optional threshold (seconds)."""
    class Timer:
        def __init__(self):
            self.threshold = None
            self.elapsed = None

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed = time.perf_counter() - self.start
            print(f"\n Call latency: {self.elapsed:.2f}s")
            if self.threshold:
                assert self.elapsed < self.threshold, (
                    f"Latency {self.elapsed:.2f}s exceeds threshold of {self.threshold}s"
                )

    def _with_threshold(threshold=None):
        t = Timer()
        t.threshold = threshold
        return t

    return _with_threshold
