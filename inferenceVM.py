from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


app = FastAPI()

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[GPU Server] Using device: {DEVICE}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

model.to(DEVICE)
model.eval()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150


def get_gpu_metrics():
    metrics = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_utilization_percent": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_temperature_c": None,
        "gpu_power_watts": None,
        "torch_memory_allocated_mb": None,
        "torch_memory_reserved_mb": None,
    }

    if DEVICE != "cuda":
        return metrics

    metrics["gpu_name"] = torch.cuda.get_device_name(0)
    metrics["torch_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024 / 1024, 2)
    metrics["torch_memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2)

    if NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        except Exception:
            power = None

        metrics["gpu_utilization_percent"] = util.gpu
        metrics["gpu_memory_used_mb"] = round(mem.used / 1024 / 1024, 2)
        metrics["gpu_memory_total_mb"] = round(mem.total / 1024 / 1024, 2)
        metrics["gpu_temperature_c"] = temp
        metrics["gpu_power_watts"] = round(power, 2) if power is not None else None

    return metrics


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "metrics": get_gpu_metrics()
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    start_time = time.time()

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    inputs = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metrics = get_gpu_metrics()

    if DEVICE == "cuda":
        metrics["torch_peak_memory_mb"] = round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 2)
    else:
        metrics["torch_peak_memory_mb"] = None

    return {
        "ok": True,
        "answer": answer,
        "model": MODEL_NAME,
        "device": DEVICE,
        "input_tokens": inputs["input_ids"].shape[-1],
        "output_tokens": generated_ids.shape[-1],
        "total_latency_seconds": round(end_time - start_time, 3),
        "metrics": metrics
    }