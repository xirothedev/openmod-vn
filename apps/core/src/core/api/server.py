from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from core.pipeline import CorePipeline

app = FastAPI(title="openmod-vn core")
_pipe: CorePipeline | None = None


def pipe() -> CorePipeline:
    global _pipe
    if _pipe is None:
        _pipe = CorePipeline()
    return _pipe


class ModerateIn(BaseModel):
    text: str


@app.post("/moderate")
def moderate(body: ModerateIn):
    return pipe().moderate(body.text).model_dump()


@app.get("/health")
def health():
    return {"ok": True}
