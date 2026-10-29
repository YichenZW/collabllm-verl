"""Serve and exercise real Qwen SFT/DPO adapters; no external API calls."""
import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from reproduction.common import RUNS, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    port = 20000 + int(job) % 30000
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    command = [sys.executable, "-m", "reproduction.serve", "--port", str(port)]
    models = ["base"]
    for mode in ["sft", "offline_dpo"]:
        checkpoint = RUNS / "checkpoints/math-hard" / f"{mode}-smoke"
        assert (checkpoint / "adapter_model.safetensors").is_file()
        models.append(f"math-{mode}-smoke")
        command += ["--adapter", f"{models[-1]}={checkpoint}"]
    with (RUNS / "logs" / f"inference-smoke-server-{job}.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError("Local inference server exited; inspect its saved log")
                try:
                    with opener.open(url + "/health", timeout=2):
                        break
                except Exception:
                    time.sleep(3)
            else:
                raise RuntimeError("Local inference server did not become healthy within five minutes")
            with opener.open(url + "/v1/models", timeout=10) as response:
                served = json.load(response)
            assert set(models).issubset({m["id"] for m in served["data"]})
            responses = {}
            for model in models:
                body = dict(model=model, temperature=0, max_tokens=32, seed=42,
                    messages=[dict(role="user", content="What is 2 + 2? Give a brief answer.")])
                request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with opener.open(request, timeout=90) as response:
                    result = json.load(response)
                assert result["choices"][0]["message"]["content"].strip()
                assert result["usage"]["completion_tokens"] > 0
                responses[model] = result
            config = json.loads((RUNS / "logs" / f"server-config-{job}-{port}.json").read_text())
            assert config["max_model_len"] == 40960
            long_responses = []
            if args.long_context:
                def long_request(index):
                    body = dict(model="base", temperature=0, max_tokens=8, seed=42 + index,
                        messages=[dict(role="user", content="Context item.\n" * 6000 +
                                       "End of context. Reply only with OK.")])
                    request = urllib.request.Request(url + "/v1/chat/completions",
                        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                    with opener.open(request, timeout=240) as response:
                        result = json.load(response)
                    assert 16000 <= result["usage"]["prompt_tokens"] < config["max_model_len"]
                    assert result["choices"][0]["message"]["content"].strip()
                    return dict(usage=result["usage"], response=result["choices"][0]["message"]["content"])
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    long_responses = list(pool.map(long_request, range(4)))
            record = dict(job=job, host=socket.gethostname(),
                native_context=config["max_model_len"], served_models=served, responses=responses,
                concurrent_long_context=long_responses,
                external_api_calls=0, scope="Inference compatibility only, not task accuracy")
            previous = RUNS / "inference_smoke.json"
            if previous.exists():
                prior = json.loads(previous.read_text())
                archive = RUNS / f"inference_smoke-{prior['job']}.json"
                if not archive.exists():
                    atomic_json(archive, prior)
            atomic_json(RUNS / f"inference_smoke-{job}.json", record)
            atomic_json(previous, record)
            print("QWEN_INFERENCE_SMOKE_PASS", models, flush=True)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    main()
