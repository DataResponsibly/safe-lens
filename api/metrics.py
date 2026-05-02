"""Prometheus metrics for the backend (GPU memory from CUDA)."""

from __future__ import annotations

import torch
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

router = APIRouter(tags=["monitoring"])

gpu_memory_utilization_ratio = Gauge(
    "safe_lens_gpu_memory_utilization_ratio",
    "Fraction of GPU device memory in use (0 through 1).",
    ["device_index"],
)


@router.get("/metrics")
def prometheus_metrics() -> Response:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_b, total_b = torch.cuda.mem_get_info(i)
            if total_b and total_b > 0:
                ratio = (total_b - free_b) / total_b
                gpu_memory_utilization_ratio.labels(device_index=str(i)).set(ratio)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
