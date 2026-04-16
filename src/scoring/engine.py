import asyncio
import json
from typing import List, Dict, Any
from src.facets.registry import FacetRegistry
from src.scoring.llm_client import LLMClient

class ScoringEngine:
    def __init__(self, registry_path: str):
        self.registry = FacetRegistry(registry_path)
        self.client = LLMClient()
        
    def _format_conversation(self, turns: List[Dict[str, str]]) -> str:
        return "\\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in turns)
        
    async def evaluate_conversation(self, conv: Dict[str, Any], facet_ids: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a conversation against a given list of facets.
        If facet_ids is None, evaluates generally or against a default subset (since 5000 is too large for memory/time).
        """
        text = self._format_conversation(conv["turns"])
        metadata = conv.get("metadata", {})
        
        all_facets = self.registry.get_all_facets()
        if facet_ids:
            facets_to_run = [f for f in all_facets if f["facet_id"] in facet_ids]
        else:
            # Default to top 10 facets for speed if none specified
            # The architecture supports scale to 5000+ by just providing all IDs, 
            # though batched scoring shouldn't call LLM 5000 times sequentially.
            facets_to_run = all_facets[:10] 
            
        tasks = []
        for facet in facets_to_run:
            tasks.append(self._eval_facet(text, facet, metadata))
            
        results = await asyncio.gather(*tasks)
        
        return {
            "conversation_id": conv["conversation_id"],
            "evaluations": {
                res["facet_id"]: res["evaluation"] for res in results
            }
        }
        
    async def _eval_facet(self, text: str, facet: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        eval_result = await self.client.evaluate_facet(text, facet, meta)
        return {
            "facet_id": facet["facet_id"],
            "evaluation": eval_result
        }

    async def evaluate_batch(self, conversations: List[Dict[str, Any]], facet_ids: List[str] = None) -> List[Dict[str, Any]]:
        # In a real system, we would rate-limit and batch LLM calls optimally.
        # Since we use asyncio and are mocking, we can trigger them all.
        tasks = [self.evaluate_conversation(conv, facet_ids) for conv in conversations]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    async def test():
        engine = ScoringEngine('facets/facet_registry.json')
        convs = []
        with open('data/processed/conversations.jsonl', 'r') as f:
            for line in f:
                convs.append(json.loads(line))
        
        if convs:
            print(f"Testing score engine with 1 conversation against 10 facets...")
            res = await engine.evaluate_conversation(convs[0])
            print(json.dumps(res, indent=2))
            
    asyncio.run(test())
