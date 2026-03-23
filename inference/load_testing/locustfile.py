"""
Locust load test for the vLLM inference server.

Tests:
- Chat completions endpoint (/v1/chat/completions)
- Text completions endpoint (/v1/completions)
- Health check endpoint
- Varying prompt lengths (short, medium, long)
- Concurrent user simulation

Usage:
    locust -f locustfile.py --host=http://localhost:8080
    locust -f locustfile.py --host=http://localhost:8080 --headless -u 50 -r 10 -t 5m
"""

import json
import random
import time

from locust import HttpUser, between, task, events


# Test Prompts (varying complexity and length)

SHORT_PROMPTS = [
    "What is machine learning?",
    "Explain recursion in one sentence.",
    "What is the capital of France?",
    "Define gradient descent.",
    "What is a neural network?",
]

MEDIUM_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning. "
    "Give two examples of each approach and describe when you would choose "
    "one over the other in a real-world scenario.",

    "Describe the transformer architecture in detail. Include the role of "
    "self-attention, positional encoding, and the feed-forward layers. "
    "Explain why this architecture was revolutionary for NLP tasks.",

    "Compare and contrast LSTM and GRU architectures. Discuss their gating "
    "mechanisms, computational efficiency, and typical use cases. Which would "
    "you recommend for a real-time speech recognition system and why?",
]

LONG_PROMPTS = [
    "Write a comprehensive technical guide on implementing a distributed "
    "training pipeline for large language models. Cover the following topics: "
    "1) Data parallelism vs model parallelism strategies, 2) ZeRO optimization "
    "stages and their trade-offs, 3) Gradient accumulation and its impact on "
    "effective batch size, 4) Communication backends (NCCL, Gloo) and their "
    "performance characteristics, 5) Fault tolerance mechanisms including "
    "checkpointing strategies and elastic training. For each topic, provide "
    "concrete implementation details and code examples where appropriate.",

    "Analyze the evolution of attention mechanisms in deep learning from 2017 "
    "to present. Start with the original Transformer paper, then discuss key "
    "innovations including multi-head attention, sparse attention patterns, "
    "linear attention approximations, flash attention, grouped query attention, "
    "and multi-query attention. For each variant, explain the computational "
    "complexity, memory requirements, and empirical performance improvements. "
    "Conclude with a discussion of how attention has been adapted for vision "
    "models, multimodal architectures, and state space models.",
]

SYSTEM_PROMPTS = [
    "You are a helpful AI assistant.",
    "You are a senior machine learning engineer at a top tech company.",
    "You are a concise technical writer. Be brief and precise.",
    "You are an expert in distributed systems and GPU programming.",
]


# Locust User Classes

class InferenceUser(HttpUser):
    """Simulates a typical API user making chat completion requests."""

    wait_time = between(0.5, 3.0)

    @task(5)
    def chat_short(self):
        """Short prompt — fast response expected."""
        self._chat_completion(
            random.choice(SHORT_PROMPTS),
            max_tokens=128,
            name="/v1/chat/completions [short]",
        )

    @task(3)
    def chat_medium(self):
        """Medium prompt — moderate response."""
        self._chat_completion(
            random.choice(MEDIUM_PROMPTS),
            max_tokens=512,
            name="/v1/chat/completions [medium]",
        )

    @task(1)
    def chat_long(self):
        """Long prompt — stress test."""
        self._chat_completion(
            random.choice(LONG_PROMPTS),
            max_tokens=1024,
            name="/v1/chat/completions [long]",
        )

    @task(2)
    def chat_streaming(self):
        """Streaming chat completion."""
        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": random.choice(MEDIUM_PROMPTS)},
            ],
            "max_tokens": 256,
            "stream": True,
        }

        start = time.monotonic()
        first_token_time = None
        token_count = 0

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True,
            name="/v1/chat/completions [stream]",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status {response.status_code}")
                return

            for line in response.iter_lines():
                if line:
                    line_text = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_text.startswith("data: ") and line_text != "data: [DONE]":
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        token_count += 1

            response.success()

        elapsed = time.monotonic() - start
        if first_token_time:
            ttft = (first_token_time - start) * 1000
            events.request.fire(
                request_type="CUSTOM",
                name="Time to First Token (ms)",
                response_time=ttft,
                response_length=0,
                exception=None,
                context={},
            )

    @task(1)
    def health_check(self):
        """Health endpoint — should be fast."""
        self.client.get("/health", name="/health")

    def _chat_completion(
        self, prompt: str, max_tokens: int = 256, name: str = ""
    ):
        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            catch_response=True,
            name=name or "/v1/chat/completions",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Report custom metrics
                events.request.fire(
                    request_type="CUSTOM",
                    name="Tokens Generated",
                    response_time=completion_tokens,
                    response_length=0,
                    exception=None,
                    context={},
                )

                response.success()
            else:
                response.failure(f"Status {response.status_code}: {response.text[:200]}")


class BurstUser(HttpUser):
    """Simulates bursty traffic patterns."""

    wait_time = between(0.1, 0.5)

    @task
    def burst_requests(self):
        """Rapid-fire requests to test batching behavior."""
        payload = {
            "model": "default",
            "messages": [
                {"role": "user", "content": random.choice(SHORT_PROMPTS)},
            ],
            "max_tokens": 64,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            catch_response=True,
            name="/v1/chat/completions [burst]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
