from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="IntentRec API", version="0.1.0")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SessionEvent(BaseModel):
    event_type: str                  # 'click', 'query', 'add_to_cart', 'purchase'
    item_id: Optional[str] = None
    query: Optional[str] = None
    timestamp: Optional[int] = None  # epoch ms


class RecommendRequest(BaseModel):
    session_id: str
    events: List[SessionEvent]
    top_k: int = 10


class RecommendResponse(BaseModel):
    session_id: str
    recommendations: List[str]
    intent_vector: Optional[List[float]] = None
    intent_shift_detected: bool = False


# ---------------------------------------------------------------------------
# Startup: load models
# ---------------------------------------------------------------------------

# TODO: add @app.on_event("startup") handler that:
#   - instantiates IntentEncoder and loads weights from disk
#   - instantiates QueryEncoder (sentence-transformers)
#   - instantiates FAISSItemIndex and calls .load() with index path
#   - stores all three in app.state so endpoints can access them
#   - loads item_vocab dict from disk


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Liveness check — returns 200 if the server is running."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Given a session's event history, return top-k item recommendations.
    """
    # TODO: raise HTTPException(400) if request.events is empty
    # TODO: extract query strings and item_ids from request.events in order
    # TODO: encode query events using app.state.query_encoder
    # TODO: look up item embeddings for click events from app.state.item_embedding
    # TODO: build a mixed input sequence tensor (queries + clicks interleaved by timestamp)
    # TODO: pass through app.state.intent_encoder → intent_vector [1, hidden_dim]
    # TODO: evaluate intent-shift detection only after validating a definition;
    #       do not assume a fixed cosine-similarity threshold
    # TODO: query app.state.faiss_index.search(intent_vector, k=request.top_k)
    # TODO: return RecommendResponse with recommendations, intent_vector.tolist(),
    #       intent_shift_detected
    pass


@app.get("/session/{session_id}/intent")
def get_session_intent(session_id: str):
    """
    Return the current intent vector for an in-progress session.
    """
    # TODO: look up session state from an in-memory cache (e.g. dict keyed by session_id)
    #       or from PostgreSQL session table
    # TODO: raise HTTPException(404, "Session not found") if session_id is unknown
    # TODO: return {"session_id": session_id, "intent_vector": <list of floats>}
    pass
