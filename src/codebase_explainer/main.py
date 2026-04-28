"""FastAPI entry point. Week-1 scaffold: a health endpoint and a placeholder /chat."""

from fastapi import FastAPI
from pydantic import BaseModel

from codebase_explainer import __version__

app = FastAPI(title="Codebase Explainer Agent", version=__version__)


class ChatRequest(BaseModel):
    repo_url: str
    question: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, str]:
    # Placeholder. Real agent loop arrives in week 3.
    return {
        "repo_url": req.repo_url,
        "question": req.question,
        "answer": "Not implemented yet — see ROADMAP in README.",
    }
