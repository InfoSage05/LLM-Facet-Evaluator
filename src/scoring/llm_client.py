import os
import httpx
import json
import re
import random
from typing import Dict, Any

class LLMClient:
    def __init__(self):
        self.use_mock = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        self.api_key = os.getenv("LLM_API_KEY", "dummy")
        self.model = os.getenv("LLM_MODEL", "qwen2:1.5b")

    async def evaluate_facet(self, conversation_text: str, facet: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single conversation against a specific facet."""
        if self.use_mock:
            return self._mock_evaluation(facet)

        prompt = self._build_prompt(conversation_text, facet, metadata)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.4}
                }

                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

                content = data["message"]["content"]

                # Greedy match to capture the full JSON object including nested braces
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if not match:
                    raise ValueError(f"No JSON object found in: {content[:200]}")

                parsed = json.loads(match.group(0))

                score = max(1, min(5, int(parsed.get("score", 3))))
                confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
                rationale = str(parsed.get("rationale", "No rationale provided."))

                return {"score": score, "confidence": confidence, "rationale": rationale}

        except Exception as e:
            print(f"  [WARN] LLM evaluation failed for facet '{facet['name']}': {e}")
            return self._mock_evaluation(facet, rationale=f"Fallback (LLM error): {e}")

    def _build_prompt(self, text: str, facet: Dict[str, Any], meta: Dict[str, Any]) -> str:
        return f"""You are an expert Conversation Analytical Grader.

Evaluate the following conversation strictly according to the grading facet.
IMPORTANT: You must output a valid JSON object. Do not output anything else.

--- Conversation Metadata ---
Turns          : {meta.get('turn_count', 'N/A')}
Total chars    : {meta.get('character_length', 'N/A')}
Avg word length: {meta.get('avg_word_length', 'N/A')}
Questions asked: {meta.get('question_count', 'N/A')}
Exclamations   : {meta.get('exclamation_count', 'N/A')}
Lexical div.   : {meta.get('lexical_diversity', 'N/A')}
Formality score: {meta.get('formality_score', 'N/A')}
Speaker balance: {meta.get('speaker_balance', 'N/A')}

--- Conversation ---
{text}

--- Facet to Evaluate ---
Name       : {facet['name']}
Group      : {facet['group']}
Description: {facet['description']}

Score Rubric:
1: {facet['rubric']['1']}
2: {facet['rubric']['2']}
3: {facet['rubric']['3']}
4: {facet['rubric']['4']}
5: {facet['rubric']['5']}

Instructions for JSON:
1. "rationale": Write a 2-3 sentence analysis explaining how the conversation aligns with the rubric.
2. "confidence": A float from 0.0 to 1.0 indicating your confidence in the score. Use strict, realistic values (e.g., 0.72, 0.83). Avoid outputting exactly 1.0. 
3. "score": An integer exactly from 1, 2, 3, 4, or 5 based purely on your rationale.

Respond EXACTLY with this JSON structure:
{{"rationale": "...", "confidence": 0.72, "score": 2}}
"""

    def _mock_evaluation(self, facet: Dict[str, Any], rationale: str = "Mock evaluation. Set USE_MOCK_LLM=false and run Ollama to use a real model.") -> Dict[str, Any]:
        """Deterministic mock fallback, used when USE_MOCK_LLM=true."""
        h = hash(facet['name']) % 5 + 1
        return {
            "score": h,
            "confidence": round(random.uniform(0.6, 0.99), 2),
            "rationale": rationale
        }
