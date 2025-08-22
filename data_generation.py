# modular_gcrs_synthetic_data.py
"""
Modular code for Step 1: Synthetic Dataset Generation
- Product catalog schema
- Conversation logs schema
- Utilities to generate and export CSV/JSONL files
"""
import csv
import json
import random
import uuid
from datetime import datetime, timedelta

# ----- Configuration -----
NUM_PRODUCTS = 100
NUM_CONVERSATIONS = 200
MESSAGES_PER_CONVERSATION = (3, 6)  # min, max messages
OUTPUT_DIR = "./synthetic_data"

# ----- Sample Templates -----
CATEGORIES = ["Electronics", "Accessories", "Audio", "Home", "Toys"]
BRANDS = ["AlphaTech", "SoundMax", "HomeEase", "PlayFun"]
ADJECTIVES = ["ultra", "pro", "max", "eco", "smart"]
NOUNS = ["Speaker", "Camera", "Headphones", "Monitor", "Keyboard"]
USER_INTENTS = [
    "I need a gift for my {adj} {noun} under ${price}.",
    "Looking for a {adj} {noun} with long battery life.",
    "Any {adj} {noun} recommendations?",
    "My friend loves {category}, what would you suggest?",
]
ASSISTANT_TEMPLATES = [
    "Sure, how about a {brand} {noun}?",
    "You might like the {brand} {noun} in category {category}.",
    "The {noun} from {brand} could be a great fit.",
]

# ----- Utility Functions -----
def random_price():
    return round(random.uniform(20.0, 500.0), 2)

def generate_products(n=NUM_PRODUCTS):
    """Generate synthetic product catalog rows."""
    products = []
    for _ in range(n):
        pid = str(uuid.uuid4())[:8]
        category = random.choice(CATEGORIES)
        brand = random.choice(BRANDS)
        adj = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        name = f"{adj.title()} {noun}"
        price = random_price()
        attributes = {
            "color": random.choice(["Black", "White", "Red", "Blue"]),
            "warranty_years": random.choice([1, 2, 3]),
        }
        description = f"A {adj} {noun} from {brand}, perfect for {category.lower()} enthusiasts."
        products.append({
            "product_id": pid,
            "product_name": name,
            "category": category,
            "brand": brand,
            "price": price,
            "attributes": json.dumps(attributes),
            "description": description,
            "image_url": f"https://example.com/img/{pid}.png"
        })
    return products


def save_catalog_csv(products, filename=f"{OUTPUT_DIR}/products.csv"):
    """Save the product catalog to CSV."""
    fieldnames = list(products[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(products)


def generate_conversations(products, n=NUM_CONVERSATIONS):
    """Generate synthetic conversation logs in JSONL format."""
    convs = []
    now = datetime.utcnow()
    for i in range(n):
        cid = f"C{1000 + i}"
        num_msgs = random.randint(*MESSAGES_PER_CONVERSATION)
        start_time = now - timedelta(days=random.randint(0, 30))
        for j in range(num_msgs):
            timestamp = (start_time + timedelta(minutes=5*j)).isoformat() + "Z"
            if j % 2 == 0:
                # user message
                intent = random.choice(USER_INTENTS).format(
                    adj=random.choice(ADJECTIVES),
                    noun=random.choice(NOUNS),
                    price=random_price(),
                    category=random.choice(CATEGORIES)
                )
                msgs = {"conversation_id": cid, "timestamp": timestamp, "speaker": "user", "message": intent}
            else:
                # assistant message
                p = random.choice(products)
                template = random.choice(ASSISTANT_TEMPLATES).format(
                    brand=p["brand"], noun=p["product_name"], category=p["category"]
                )
                msgs = {"conversation_id": cid, "timestamp": timestamp, "speaker": "assistant", "message": template}
            convs.append(msgs)
    return convs


def save_conversations_jsonl(convs, filename=f"{OUTPUT_DIR}/conversations.jsonl"):
    """Save conversation logs to JSONL file."""
    with open(filename, "w") as f:
        for msg in convs:
            f.write(json.dumps(msg) + "\n")


# ----- Main Execution -----
if __name__ == "__main__":
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    products = generate_products()
    save_catalog_csv(products)
    convs = generate_conversations(products)
    save_conversations_jsonl(convs)
    print(f"Generated {len(products)} products and {len(convs)} messages in {OUTPUT_DIR}")
