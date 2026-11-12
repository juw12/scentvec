"""Minimal tour of the scentvec Python API.

    ./.venv/bin/python examples/quickstart.py
"""

from __future__ import annotations

from scentvec.engine import ScentEngine

engine = ScentEngine.load()

# 1. Find a perfume by name, then its nearest neighbours.
aventus = engine.lookup("creed aventus", limit=1)[0]
neighbours = engine.similar(aventus["item_id"], k=5, max_per_brand=1)
print(f"Similar to {aventus['brand']} - {aventus['name']}:")
for row in neighbours["results"]:
    print(f"  {row['score']:.3f}  {row['brand']} - {row['name']}")

# 2. Search by description.
found = engine.search(text="smoky vanilla and tobacco, not too sweet, no patchouli", k=5)
print("\nSmoky vanilla tobacco, no patchouli:")
for row in found["results"]:
    print(f"  {row['score']:.3f}  {row['brand']} - {row['name']}")

# 3. Embed a note pyramid that is not in the catalog.
vector = engine.encoder.embed(
    top=["bergamot", "lemon"],
    middle=["rose"],
    base=["musk", "cedarwood"],
)
print(f"\nEmbedding: shape {vector.shape}, norm {float((vector ** 2).sum()) ** 0.5:.3f}")

# 4. Why are two perfumes close?
closest = neighbours["results"][0]
reasons = engine.explain(aventus["item_id"], closest["item_id"])
print(f"\nShared notes with {closest['name']}: {', '.join(reasons['shared_notes'])}")
