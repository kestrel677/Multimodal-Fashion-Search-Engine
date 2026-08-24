"""Multimodal fashion search API backed by CLIP and FAISS."""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashion_search")

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "openai/clip-vit-base-patch32"
INDEX_PATH = BASE_DIR / "fashion_faiss.index"
METADATA_PATH = BASE_DIR / "processed_styles.csv"
STATIC_DIR = BASE_DIR / "static"
MAX_TOP_K = 100
MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_URL_TEMPLATE = "/static/images/{item_id}.jpg"
MODEL_LOCAL_ONLY = os.getenv("MODEL_LOCAL_ONLY", "false").lower() in {"1", "true", "yes"}


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1_000)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


class SearchResources:
    """Objects loaded once at startup and shared by request handlers."""

    def __init__(self, model: CLIPModel, processor: CLIPProcessor, index: Any, metadata: pd.DataFrame, device: torch.device) -> None:
        self.model = model
        self.processor = processor
        self.index = index
        self.metadata = metadata
        self.device = device


def load_resources() -> SearchResources:
    """Load and validate all local and remote inference resources."""
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading CLIP model %s on %s", MODEL_NAME, device)
    try:
        processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=MODEL_LOCAL_ONLY)
        model = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=MODEL_LOCAL_ONLY).to(device)
    except OSError as exc:
        mode = "local cache" if MODEL_LOCAL_ONLY else "Hugging Face or the local cache"
        raise RuntimeError(
            f"Unable to load {MODEL_NAME} from {mode}. "
            "Allow access to https://huggingface.co once to download the model, "
            "or pre-download it into the Hugging Face cache."
        ) from exc
    model.eval()

    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)
    if index.d != 512:
        raise ValueError(f"Expected a 512-dimensional FAISS index, got {index.d}")
    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata size mismatch: index has {index.ntotal} vectors, "
            f"metadata has {len(metadata)} rows"
        )

    logger.info("Loaded %d indexed fashion products", index.ntotal)
    return SearchResources(model, processor, index, metadata, device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep the API alive if an external model download is unavailable. Search endpoints
    # then return a useful 503 response instead of the host showing a generic 500.
    app.state.resources = None
    app.state.resource_error = None
    try:
        app.state.resources = await run_in_threadpool(load_resources)
    except Exception as exc:
        app.state.resource_error = str(exc)
        logger.exception("Search resources could not be loaded")
    yield


app = FastAPI(title="Multimodal Fashion Search API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def get_resources(request: Request) -> SearchResources:
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        error = getattr(request.app.state, "resource_error", None)
        detail = "Search service is unavailable because its model resources did not load"
        if error:
            detail = f"{detail}: {error}"
        raise HTTPException(status_code=503, detail=detail)
    return resources


def normalize_embedding(embedding: torch.Tensor | Any) -> np.ndarray:
    """Convert CLIP's output into one FAISS-ready, L2-normalized float32 vector."""
    # transformers 4 returns a Tensor here; transformers 5 returns a
    # BaseModelOutputWithPooling whose pooler_output is the projected feature.
    if not isinstance(embedding, torch.Tensor):
        embedding = getattr(embedding, "pooler_output", None)
    if not isinstance(embedding, torch.Tensor):
        raise TypeError("CLIP did not return a tensor embedding")
    vector = embedding.detach().cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    if vector.shape != (1, 512):
        raise RuntimeError(f"Expected a (1, 512) embedding, got {vector.shape}")
    return vector


def encode_text(query: str, resources: SearchResources) -> np.ndarray:
    inputs = resources.processor(text=[query], return_tensors="pt", padding=True)
    inputs = {name: value.to(resources.device) for name, value in inputs.items()}
    with torch.inference_mode():
        return normalize_embedding(resources.model.get_text_features(**inputs))


def encode_image(image: Image.Image, resources: SearchResources) -> np.ndarray:
    inputs = resources.processor(images=[image], return_tensors="pt")
    inputs = {name: value.to(resources.device) for name, value in inputs.items()}
    with torch.inference_mode():
        return normalize_embedding(resources.model.get_image_features(**inputs))

def encode_multimodal(image: Image.Image, text: str, resources: SearchResources) -> np.ndarray:
    """Combine image and text features into a single normalized embedding."""
    img_vector = encode_image(image, resources)
    txt_vector = encode_text(text, resources)
    combined = (img_vector + txt_vector) / 2.0
    faiss.normalize_L2(combined)
    return combined

def search(embedding: np.ndarray, top_k: int, resources: SearchResources) -> list[dict[str, Any]]:
    distances, indices = resources.index.search(embedding, top_k)
    results: list[dict[str, Any]] = []
    for rank, (distance, row_index) in enumerate(zip(distances[0], indices[0]), start=1):
        if row_index < 0:
            continue
        # Convert NumPy/Pandas scalar values and missing values into JSON-safe objects.
        product = resources.metadata.iloc[int(row_index)].where(pd.notna(resources.metadata.iloc[int(row_index)]), None).to_dict()
        product = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in product.items()}
        # Serve product images from the public dataset repository rather than
        # relying on machine-specific local Kaggle paths.
        item_id = product.get("id")
        if item_id is not None:
            formatted_id = str(int(float(item_id)))
            image_url = IMAGE_URL_TEMPLATE.format(item_id=formatted_id)
        else:
            image_url = None
        product["image_url"] = image_url
        product["image_path"] = image_url
        product.update(
            {
                "rank": rank,
                "distance": float(distance),
                "similarity_score": float(distance),
            }
        )
        results.append(product)
    return results


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("index.html", media_type="text/html")


async def run_text_search(query: str, top_k: int, resources: SearchResources) -> dict[str, list[dict[str, Any]]]:
    embedding = await run_in_threadpool(encode_text, query.strip(), resources)
    return {"results": await run_in_threadpool(search, embedding, top_k, resources)}


@app.post("/text-search")
async def text_search(
    query: str = Query(..., min_length=1, max_length=1_000),
    top_k: int = Query(default=5, ge=1, le=MAX_TOP_K),
    resources: SearchResources = Depends(get_resources),
) -> dict[str, list[dict[str, Any]]]:
    return await run_text_search(query, top_k, resources)


@app.post("/api/search/text")
async def legacy_text_search(
    payload: TextSearchRequest, resources: SearchResources = Depends(get_resources)
) -> dict[str, list[dict[str, Any]]]:
    return await run_text_search(payload.query, payload.top_k, resources)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)