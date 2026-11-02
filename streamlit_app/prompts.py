"""Zero-shot CLIP prompts used as semantic priors (must match Reproducibility/feature_extraction_signature.json)."""
PROMPTS = [
    "a person hiding an item inside their pocket or jacket",
    "a person putting retail merchandise into a backpack or bag",
    "a person concealing an object under their shirt",
    "a normal customer browsing shelves and holding an item normally",
]
PROMPT_LABELS = ["Pocket / jacket conceal", "Bag / backpack stuffing", "Under-shirt conceal", "Normal browsing"]
