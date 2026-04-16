from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json

# Load .env from project root
_env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from src.scoring.engine import ScoringEngine
from src.facets.registry import FacetRegistry
from src.preprocess.preprocessor import Conversation, Preprocessor

app = FastAPI(title="Facet Evaluation API", version="1.0.0")

# Using a global engine instance
REGISTRY_PATH = os.getenv("FACET_REGISTRY", "facets/facet_registry.json")
engine = ScoringEngine(REGISTRY_PATH)
preprocessor = Preprocessor()

class ScoreRequest(BaseModel):
    conversation: Conversation
    facet_ids: Optional[List[str]] = None

class BatchScoreRequest(BaseModel):
    conversations: List[Conversation]
    facet_ids: Optional[List[str]] = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/facets")
def get_facets(group: Optional[str] = None):
    if group:
        return engine.registry.get_facets_by_group(group)
    return engine.registry.get_all_facets()

@app.post("/score")
async def score_conversation(request: ScoreRequest):
    try:
        # Preprocess
        processed_conv = preprocessor.extract_features(request.conversation)
        conv_dict = processed_conv.model_dump()
        
        # Evaluate
        result = await engine.evaluate_conversation(conv_dict, request.facet_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_score")
async def batch_score(request: BatchScoreRequest):
    try:
        results = []
        # Preprocess all
        for conv in request.conversations:
            processed = preprocessor.extract_features(conv)
            results.append(processed.model_dump())
            
        evaluation = await engine.evaluate_batch(results, request.facet_ids)
        return {"results": evaluation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
