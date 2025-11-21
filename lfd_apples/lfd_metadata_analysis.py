import os
import json
from collections import Counter
from tqdm import tqdm


def collect_json_files(base_dir):
    # --------------------------------------
    # STEP 1 — collect all JSON file paths
    # --------------------------------------
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        # Optional: remove weird system folders
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    print(f"Found {len(json_files)} JSON files")

    return json_files


def complete_json_file(json_files):
    # --------------------------------------
    # STEP 2 — check completeness of JSON files
    # --------------------------------------
    incomplete_files = []
    print('')
    for json_path in tqdm(json_files, desc="Checking JSON completeness"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Check for required fields
            proxy = data.get("proxy", {})
            apple = proxy.get("apple", {})
            spur = proxy.get("spur", {})
            stem = proxy.get("stem", {})

            # Check apple id
            if "id" not in apple: 
                print(f"Found APPLE case {json_path}")
                incomplete_files.append(json_path)

                # Case 1: All initial trials performed with apple with mass 173g
                apple_mass = apple.get("mass")
                apple_diameter = apple.get("diameter")  
                apple_height = apple.get("height")

                if apple_mass == "173 g":
                    print(f"Found apple case 1 in {json_path}")

                    # Assign an ID — adjust as needed
                    apple["id"] = "1"

                    # Write back into main structure
                    proxy["apple"] = apple
                    data["proxy"] = proxy

                    # Save JSON back to file
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=4)

                if apple_mass == "254 g" and apple_diameter == "80 mm" and apple_height == "80 mm":
                    print(f"Found apple case 2 in {json_path}")  

                    # Assign an ID — adjust as needed
                    apple["id"] = "3"

                    # Write back into main structure
                    proxy["apple"] = apple
                    data["proxy"] = proxy

                    # Save JSON back to file
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=4)

            # Check spur id
            if "id" not in spur:
                print(f"Found missing SPUR case {json_path}")
                incomplete_files.append(json_path)

                spur_diam = spur.get("diameter")
                spur_length = spur.get("length")
                stem_magnet = stem.get("magnet")


                # SPUR ID 7
                if spur_diam == "12 mm" and spur_length == "40 mm" and stem_magnet == "medium":
                    print(f"Found case 2 in {json_path}")

                    # Assign an ID — adjust as needed
                    spur["id"] = "7"

                    # Write back into main structure
                    proxy["spur"] = spur
                    data["proxy"] = proxy

                    # Save JSON back to file
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=4)
                
                # SPUR ID 4
                if spur_diam == "5 mm" and spur_length == "60 mm" and stem_magnet == "medium":
                    print(f"Found case 3 in {json_path}")

                    # Assign an ID — adjust as needed
                    spur["id"] = "4"

                    # Write back into main structure
                    proxy["spur"] = spur
                    data["proxy"] = proxy

                    # Save JSON back to file
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=4)



        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            incomplete_files.append(json_path)

    print(f"Found {len(incomplete_files)} incomplete JSON files")

    print("Incomplete files:")
    # for f in incomplete_files:
        # print(f )

    return incomplete_files




def count_trials_variations(json_files):

    apple_counter = Counter()
    spur_counter = Counter()

    # --------------------------------------
    # STEP 2 — process with tqdm
    # --------------------------------------
    for json_path in tqdm(json_files, desc="Processing trials"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Get fields inside proxy
            proxy = data.get("proxy", {})

            apple = proxy.get("apple", {})
            spur = proxy.get("spur", {})

            apple_id = apple.get("id")
            spur_id  = spur.get("id")

            if apple_id is not None:
                apple_counter[str(apple_id)] += 1

            if spur_id is not None:
                spur_counter[str(spur_id)] += 1

        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    print("\n=== Apple Counts ===")
    for apple, count in sorted(apple_counter.items(), key=lambda x: int(x[0])):
        print(f"Apple {apple}: {count}")

    print("\n=== Spur Counts ===")
    for spur, count in sorted(spur_counter.items(), key=lambda x: int(x[0])):
        print(f"Spur {spur}: {count}")


def main():

    # Path to your main directory
    BASE_DIR = "/media/guest/IL_data/01_IL_bagfiles"

    json_files = collect_json_files(BASE_DIR)
    count_trials_variations(json_files)
    complete_json_file(json_files)


if __name__ == '__main__':
    main()
