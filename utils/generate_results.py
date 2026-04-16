import asyncio
import json
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scoring.engine import ScoringEngine

# --- Load .env variables if present ---
env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

async def score_one(engine, conv, facets_to_run):
    """Score a single conversation sequentially (facet by facet) to avoid overloading the model."""
    all_facets = engine.registry.get_all_facets()
    facet_map = {f['facet_id']: f for f in all_facets}

    evaluations = {}
    text = engine._format_conversation(conv["turns"])
    metadata = conv.get("metadata", {})

    for fid in facets_to_run:
        facet = facet_map.get(fid)
        if not facet:
            continue
        result = await engine.client.evaluate_facet(text, facet, metadata)
        evaluations[fid] = result

    return {
        "conversation_id": conv["conversation_id"],
        "evaluations": evaluations
    }

async def main():
    use_mock = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
    model = os.getenv("LLM_MODEL", "qwen2:1.5b")
    print(f"{'[MOCK MODE]' if use_mock else f'[LIVE MODEL: {model}]'} Loading conversations...")

    conversations = []
    with open('data/processed/conversations.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Loaded {len(conversations)} conversations.")

    engine = ScoringEngine('facets/facet_registry.json')
    all_facets = engine.registry.get_all_facets()
    # Score 10 facets per conversation for speed — this is 500 real LLM calls total
    facets_to_run = [f['facet_id'] for f in all_facets[:10]]
    print(f"Scoring against {len(facets_to_run)} facets per conversation (sequential mode)...\n")

    os.makedirs('outputs', exist_ok=True)
    out_file = 'outputs/scored_conversations.jsonl'

    total = len(conversations)
    with open(out_file, 'w', encoding='utf-8') as f:
        for idx, conv in enumerate(conversations, 1):
            t0 = time.time()
            result = await score_one(engine, conv, facets_to_run)
            elapsed = time.time() - t0
            f.write(json.dumps(result) + '\n')
            f.flush()

            sample_facet = list(result['evaluations'].keys())[0]
            sample_eval = result['evaluations'][sample_facet]
            print(f"  [{idx:02d}/{total}] {conv['conversation_id']} | "
                  f"facet='{sample_facet}' score={sample_eval['score']} "
                  f"conf={sample_eval['confidence']:.2f} ({elapsed:.1f}s)")

    print(f"\n[DONE] Scored {total} conversations x {len(facets_to_run)} facets -> {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
