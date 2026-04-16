import json
import re
from typing import List, Dict, Any
from pydantic import BaseModel

FORMAL_MARKERS = {"please", "thank", "sir", "madam", "however", "therefore", "sincerely", "regards", "appreciate", "apologies"}

class Turn(BaseModel):
    role: str
    content: str
    
class Conversation(BaseModel):
    conversation_id: str
    scenario: str
    description: str
    turns: List[Turn]
    metadata: Dict[str, Any] = {}

class Preprocessor:
    def __init__(self):
        pass

    def extract_features(self, conv: Conversation) -> Conversation:
        all_words = [w for turn in conv.turns for w in turn.content.split()]
        token_count = len(all_words)
        turn_count = len(conv.turns)
        roles = list(set(turn.role for turn in conv.turns))
        length = sum(len(turn.content) for turn in conv.turns)

        avg_word_len = sum(len(w) for w in all_words) / max(token_count, 1)
        question_count = sum(turn.content.count('?') for turn in conv.turns)
        exclamation_count = sum(turn.content.count('!') for turn in conv.turns)
        longest_turn_length = max((len(turn.content) for turn in conv.turns), default=0)
        avg_turn_length = length / max(turn_count, 1)

        unique_words = set(w.lower().strip(".,!?\"'") for w in all_words)
        lexical_diversity = round(len(unique_words) / max(token_count, 1), 4)

        user_words = sum(len(t.content.split()) for t in conv.turns if t.role.lower() in ("user", "human"))
        assistant_words = sum(len(t.content.split()) for t in conv.turns if t.role.lower() in ("assistant", "ai", "bot"))
        total_spoken = user_words + assistant_words
        speaker_balance = round(user_words / max(total_spoken, 1), 4)

        formal_hits = sum(1 for w in all_words if w.lower().strip(".,!?\"'") in FORMAL_MARKERS)
        formality_score = round((formal_hits / max(token_count, 1)) * 100, 4)

        toxicity_signal = 0.0
        sentiment_signal = 0.5

        conv.metadata.update({
            "token_count": token_count,
            "turn_count": turn_count,
            "speaker_roles": roles,
            "character_length": length,
            "avg_word_length": round(avg_word_len, 2),
            "avg_turn_length": round(avg_turn_length, 2),
            "question_count": question_count,
            "exclamation_count": exclamation_count,
            "longest_turn_length": longest_turn_length,
            "lexical_diversity": lexical_diversity,
            "speaker_balance": speaker_balance,
            "formality_score": formality_score,
            "language_detected": "en",
            "toxicity_signal": toxicity_signal,
            "sentiment_signal": sentiment_signal,
        })
        return conv

def process_conversations(input_filepath: str, output_filepath: str):
    preprocessor = Preprocessor()
    processed = []
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            conv = Conversation(**data)
            conv = preprocessor.extract_features(conv)
            processed.append(conv)
            
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for p in processed:
            f.write(p.model_dump_json() + '\n')
    
    print(f"[OK] Processed {len(processed)} conversations -> {output_filepath}")
    # Print feature summary for first record
    if processed:
        meta = processed[0].metadata
        print("\nSample feature snapshot (conversation 1):")
        for k, v in meta.items():
            print(f"  {k:25s}: {v}")

if __name__ == "__main__":
    import os
    os.makedirs('data/processed', exist_ok=True)
    process_conversations('data/raw/conversations.jsonl', 'data/processed/conversations.jsonl')
