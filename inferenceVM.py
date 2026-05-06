from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

model.to(DEVICE)
model.eval()


class BatchItem(BaseModel):
    task_id: int
    prompt: str


class GenerateRequest(BaseModel):
    requests: List[BatchItem]
    max_new_tokens: int = 150


def get_gpu_utilization():
    if DEVICE != "cuda" or not NVML_AVAILABLE:
        return None

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "gpu_utilization": get_gpu_utilization()
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        prompts = [item.prompt for item in req.requests]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        prompt_length = inputs["input_ids"].shape[1]
        responses = []

        for index, item in enumerate(req.requests):
            generated_ids = output_ids[index][prompt_length:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append({
                "task_id": item.task_id,
                "answer": answer
            })

        return {
            "ok": True,
            "responses": responses,
            "gpu_utilization": get_gpu_utilization()
        }
    except Exception as e:
        return {
            "ok": False,
            "responses": [
                {
                    "task_id": item.task_id,
                    "error": str(e)
                }
                for item in req.requests
            ],
            "gpu_utilization": get_gpu_utilization(),
            "error": str(e)
        }
