import os
import json
from collections import Counter


# Path to your main directory
BASE_DIR = "/media/guest/IL_data/01_IL_bagfiles"

apple_counter = Counter()
spur_counter = Counter()

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(root, file)

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                # Extract apple and spur IDs if present
                apple_id = None
                spur_id = None

                proxy = data.get("proxy", {})

                apple = proxy.get("apple", {})
                spur = proxy.get("spur", {})

                apple_id = apple.get("id")
                spur_id = spur.get("id")
               

                # Count if valid
                if apple_id is not None:
                    apple_counter[str(apple_id)] += 1

                if spur_id is not None:
                    spur_counter[str(spur_id)] += 1

            except Exception as e:
                print(f"Error reading {json_path}: {e}")

print("=== Apple Counts ===")
for apple, count in sorted(apple_counter.items(), key=lambda x: int(x[0])):
    print(f"Apple {apple}: {count}")

print("\n=== Spur Counts ===")
for spur, count in sorted(spur_counter.items(), key=lambda x: int(x[0])):
    print(f"Spur {spur}: {count}")
