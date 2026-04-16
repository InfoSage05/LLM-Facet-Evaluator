import json
import random
import csv
import os

# Create conversations
scenarios = [
    ("tech_support", "Customer having issue with software."),
    ("medical_query", "Patient asking about symptoms."),
    ("chit_chat", "Friends talking about a recent movie."),
    ("venting", "User frustrated about their day."),
    ("code_debugging", "Developer asking for help with Python."),
    ("sales", "Agent trying to sell a subscription."),
    ("argument", "Two people debating a controversial topic."),
    ("information_retrieval", "User asking for historical facts."),
    ("creative_writing", "User asking AI to write a poem."),
    ("travel_booking", "User booking a hotel and flight.")
]

conversations = []
for i in range(50):
    scenario, desc = random.choice(scenarios)
    num_turns = random.randint(2, 8)
    turns = []
    for t in range(num_turns):
        role = "user" if t % 2 == 0 else "assistant"
        text = f"[{scenario}] This is a dummy generated text for turn {t+1} representing a {role} utterance."
        
        # Add some variety to text length
        if random.random() > 0.5:
            text += " " + " ".join(["blah"] * random.randint(5, 20))
            
        turns.append({"role": role, "content": text})
    
    conversations.append({
        "conversation_id": f"conv_{i:03d}",
        "scenario": scenario,
        "description": desc,
        "turns": turns
    })

os.makedirs('data/raw', exist_ok=True)
with open('data/raw/conversations.jsonl', 'w', encoding='utf-8') as f:
    for conv in conversations:
        f.write(json.dumps(conv) + '\n')

print(f"Generated {len(conversations)} conversations in data/raw/conversations.jsonl")

# Process facets
facets_csv_path = 'data/raw/facets.csv'
facets = []

if os.path.exists(facets_csv_path):
    with open(facets_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for idx, row in enumerate(reader):
            if not row: continue
            facet_name = row[0].strip()
            # Assign random groups for demonstration
            group = random.choice(["Linguistics", "Pragmatics", "Safety", "Emotion", "Cognitive", "Social", "Clinical"])
            
            facets.append({
                "facet_id": f"F{idx:04d}",
                "name": facet_name,
                "group": group,
                "description": f"Evaluation of {facet_name} in the conversation.",
                "rubric": {
                    "1": f"Very low {facet_name}",
                    "2": f"Low {facet_name}",
                    "3": f"Moderate {facet_name}",
                    "4": f"High {facet_name}",
                    "5": f"Very high {facet_name}"
                }
            })

os.makedirs('facets', exist_ok=True)
with open('facets/facet_registry.json', 'w', encoding='utf-8') as f:
    json.dump({"facets": facets}, f, indent=2)
print(f"Processed {len(facets)} facets into facets/facet_registry.json")
