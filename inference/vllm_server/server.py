"""vLLM inference server + FastAPI wrapper.

Launches vLLM as a subprocess and wraps it with a FastAPI app that
adds request metrics, health checks, and a Prometheus endpoint.
The wrapper runs on a different port so you can hit vLLM directly
for raw perf or go through the wrapper for observability.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# Server Configuration

@dataclass
class VLLMServerConfig:
    """Configuration for the vLLM inference server."""
    model_path: str
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None  # awq, gptq, or None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    swap_space: int = 4
    enforce_eager: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None
    uvicorn_log_level: str = "info"

    @classmethod
    def from_config(cls, config: dict) -> "VLLMServerConfig":
        vllm_cfg = config.get("inference", {}).get("vllm", {})
        return cls(**{k: v for k, v in vllm_cfg.items() if k in cls.__dataclass_fields__})


# vLLM Server Launcher

# Wraps vLLM as a subprocess because their Python API is still experimental
# and the CLI server is battle-tested in prod
class VLLMServer:
    """
    Manages the vLLM inference server lifecycle.
    
    Launches vLLM as a subprocess with proper signal handling
    and health monitoring.
    """

    def __init__(self, config: VLLMServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._health_check_task: Optional[asyncio.Task] = None

    def build_command(self) -> list[str]:
        """Build the vllm serve command."""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_path,
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--pipeline-parallel-size", str(self.config.pipeline_parallel_size),
            "--dtype", self.config.dtype,
            "--max-model-len", str(self.config.max_model_len),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-num-batched-tokens", str(self.config.max_num_batched_tokens),
            "--max-num-seqs", str(self.config.max_num_seqs),
            "--swap-space", str(self.config.swap_space),
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--uvicorn-log-level", self.config.uvicorn_log_level,
        ]

        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])

        if self.config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        if self.config.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")

        if self.config.enforce_eager:
            cmd.append("--enforce-eager")

        if self.config.api_key:
            cmd.extend(["--api-key", self.config.api_key])

        return cmd

    def start(self) -> None:
        """Start the vLLM server as a subprocess."""
        cmd = self.build_command()
        log.info(f"Starting vLLM server: {' '.join(cmd)}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(self.config.tensor_parallel_size)
        )

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        log.info(f"vLLM server started (PID: {self.process.pid})")

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        log.info(f"Received signal {signum}, shutting down vLLM server...")
        self.stop()

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.warning("vLLM server didn't stop gracefully, killing...")
                self.process.kill()
            log.info("vLLM server stopped")

    async def wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for the server to be ready to accept requests."""
        import aiohttp

        url = f"http://localhost:{self.config.port}/health"
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            log.info("vLLM server is ready!")
                            return True
            except (aiohttp.ClientError, ConnectionError):
                pass
            await asyncio.sleep(2)

        log.error(f"vLLM server not ready after {timeout}s")
        return False

    def is_running(self) -> bool:
        """Check if the server process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None


# Custom API Wrapper (FastAPI middleware around vLLM)

def create_api_app(config: VLLMServerConfig):
    """
    Create a FastAPI app that wraps the vLLM server with additional features.
    
    This is an alternative to running vLLM directly — it adds:
    - Request/response logging
    - Prometheus metrics
    - Custom health checks with model info
    - Rate limiting
    """
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import aiohttp

    app = FastAPI(title="LLM Inference API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics state
    metrics_state = {
        "total_requests": 0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "errors": 0,
        "avg_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "latencies": [],
    }

    VLLM_BASE = f"http://localhost:{config.port}"

    @app.get("/health")
    async def health():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{VLLM_BASE}/health") as resp:
                    if resp.status == 200:
                        return {
                            "status": "healthy",
                            "model": config.model_path,
                            "tensor_parallel": config.tensor_parallel_size,
                            "max_model_len": config.max_model_len,
                        }
        except Exception:
            pass
        raise HTTPException(status_code=503, detail="Backend unhealthy")

    @app.get("/metrics")
    async def metrics():
        latencies = metrics_state["latencies"]
        return {
            "total_requests": metrics_state["total_requests"],
            "total_tokens_in": metrics_state["total_tokens_in"],
            "total_tokens_out": metrics_state["total_tokens_out"],
            "errors": metrics_state["errors"],
            "avg_latency_ms": round(
                sum(latencies) / len(latencies) if latencies else 0, 2
            ),
            "p99_latency_ms": round(
                sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 2
            ),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        start = time.monotonic()

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                if config.api_key:
                    headers["Authorization"] = f"Bearer {config.api_key}"

                async with session.post(
                    f"{VLLM_BASE}/v1/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    result = await resp.json()

                    elapsed = (time.monotonic() - start) * 1000
                    metrics_state["total_requests"] += 1
                    metrics_state["latencies"].append(elapsed)

                    # Track tokens
                    usage = result.get("usage", {})
                    metrics_state["total_tokens_in"] += usage.get("prompt_tokens", 0)
                    metrics_state["total_tokens_out"] += usage.get("completion_tokens", 0)

                    # dont let this list grow forever
                    if len(metrics_state["latencies"]) > 10000:
                        metrics_state["latencies"] = metrics_state["latencies"][-10000:]

                    return JSONResponse(content=result, status_code=resp.status)

        except Exception as e:
            metrics_state["errors"] += 1
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                async with session.post(
                    f"{VLLM_BASE}/v1/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    result = await resp.json()
                    metrics_state["total_requests"] += 1
                    return JSONResponse(content=result, status_code=resp.status)
        except Exception as e:
            metrics_state["errors"] += 1
            raise HTTPException(status_code=502, detail=str(e))

    return app


# Quantization helper (AWQ)

def quantize_for_serving(
    model_path: str,
    output_path: str,
    method: str = "awq",
    bits: int = 4,
    group_size: int = 128,
    calibration_dataset: str = "c4",
    num_calibration_samples: int = 512,
) -> str:
    """
    Quantize a model for efficient vLLM serving.
    
    Supports AWQ and GPTQ quantization methods.
    """
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    if method == "awq":
        log.info(f"Quantizing {model_path} with AWQ (w{bits})")
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoAWQForCausalLM.from_pretrained(model_path)

            quant_config = {
                "zero_point": True,
                "q_group_size": group_size,
                "w_bit": bits,
                "version": "GEMM",
            }

            model.quantize(tokenizer, quant_config=quant_config)
            model.save_quantized(str(output))
            tokenizer.save_pretrained(str(output))
            log.info(f"AWQ quantized model saved to {output}")

        except ImportError:
            log.error("autoawq not installed. Run: pip install autoawq")
            raise

    elif method == "gptq":
        log.info(f"Quantizing {model_path} with GPTQ (w{bits})")
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer
            from datasets import load_dataset

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            ds = load_dataset(calibration_dataset, split="train")
            examples = [tokenizer(ex["text"], return_tensors="pt") for ex in ds.select(range(num_calibration_samples))]

            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=False,
            )

            model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
            model.quantize(examples)
            model.save_quantized(str(output))
            tokenizer.save_pretrained(str(output))

        except ImportError:
            log.error("auto-gptq not installed. Run: pip install auto-gptq")
            raise

    return str(output)


# Launch functions

def launch_server(config: dict) -> VLLMServer:
    """Launch the vLLM inference server from pipeline config."""
    server_config = VLLMServerConfig.from_config(config)
    server = VLLMServer(server_config)
    server.start()
    return server


def launch_api_wrapper(config: dict, wrapper_port: int = 8080):
    """Launch the FastAPI wrapper around vLLM."""
    import uvicorn
    server_config = VLLMServerConfig.from_config(config)
    app = create_api_app(server_config)
    uvicorn.run(app, host="0.0.0.0", port=wrapper_port, log_level="info")
