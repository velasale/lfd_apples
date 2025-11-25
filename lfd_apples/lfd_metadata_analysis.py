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

            apple_mass = apple.get("mass")
            apple_diameter = apple.get("diameter")  
            apple_height = apple.get("height")

            spur_diam = spur.get("diameter")
            spur_length = spur.get("length")
            stem_magnet = stem.get("magnet")


            # Check apple id
            if "id" not in apple: 
                print(f"Found APPLE case {json_path}")
                incomplete_files.append(json_path)

                # Case 1: All initial trials performed with apple with mass 173g
                

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

                # SPUR ID 7
                if spur_diam == "12 mm" and spur_length == "40 mm" and stem_magnet == "medium":
                    print(f"Found stem case 1 in {json_path}")

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
                    print(f"Found stem case 2 in {json_path}")

                    # Assign an ID — adjust as needed
                    spur["id"] = "4"

                    # Write back into main structure
                    proxy["spur"] = spur
                    data["proxy"] = proxy

                    # Save JSON back to file
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=4)

                # SPUR ID 1
                if spur_diam == "10 mm" and spur_length == "25 mm" and stem_magnet == "medium":
                    print(f"Found stem case 3 in {json_path}")

                    # Assign an ID — adjust as needed
                    spur["id"] = "7"
                    spur["diameter"] = "12 mm"
                    spur["length"] = "40 mm"

                    # Write back into main structure
                    proxy["spur"] = spur
                    data["proxy"] = proxy

                    # Save JSON back to file
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=4)

            # Fix spur id
            if spur_diam == "12 mm" and spur_length == "40 mm" and stem_magnet == "medium" and spur.get("id") != "7":
                print(f"Correcting spur ID to 7 in {json_path}")

                # Assign correct ID
                spur["id"] = "7"

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


def count_apple_proxy_variations(json_files):

    # Open reference apple_proxy.json to get all possible variations
    with open("lfd_apples/data/apple_proxy.json", "r") as ap_prox:
        reference_data = json.load(ap_prox)
    spurs_ref = reference_data.get("spurs", {})
    apples_ref = reference_data.get("apples", {})

    apple_counter = Counter()
    spur_counter = Counter()

    apple_colors = Counter()
    stem_magnets = Counter()
    stiffness_scales = Counter()

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

            apple_color = apples_ref.get(str("id_" + apple_id)).get("color")
            stem_magnet = apples_ref.get(str("id_" + apple_id)).get("stem magnet")
            apple_colors[apple_color] += 1
            stem_magnets[stem_magnet] += 1
            stiffness_scale = spurs_ref.get(str("id_" + spur_id)).get("stiffness-scale")    
            stiffness_scales[stiffness_scale] += 1

            if apple_id is not None:
                apple_counter[str(apple_id)] += 1

            if spur_id is not None:
                spur_counter[str(spur_id)] += 1

        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    # --------------------------------------
    # APPLE STATS
    # --------------------------------------
    total_apples = sum(apple_counter.values())
    print("\n=== Apple Counts ===: ", total_apples)
    for apple, count in sorted(apple_counter.items(), key=lambda x: int(x[0])):
        pct = (count / total_apples) * 100 if total_apples > 0 else 0
        print(f"Apple {apple}: {count} ({pct:.1f}%)")
    print("\n=== Apple Color Counts ===")
    for color, count in sorted(apple_colors.items(), key=lambda x: x[0]):
        pct = (count / total_apples) * 100 if total_apples > 0 else 0
        print(f"Color {color}: {count} ({pct:.1f}%)")   
    print("\n=== Stem Magnet Counts ===")
    for magnet, count in sorted(stem_magnets.items(), key=lambda x: x[0]):
        pct = (count / total_apples) * 100 if total_apples > 0 else 0
        print(f"Stem Magnet {magnet}: {count} ({pct:.1f}%)")

    # --------------------------------------
    # SPUR STATS
    # --------------------------------------
    total_spurs = sum(spur_counter.values())
    print("\n=== Spur Counts ===: ", total_spurs)
    for spur, count in sorted(spur_counter.items(), key=lambda x: int(x[0])):
        pct = (count / total_spurs) * 100 if total_spurs > 0 else 0
        print(f"Spur {spur}: {count} ({pct:.1f}%)")
    print("\n=== Stiffness Scale Counts ===")
    for scale, count in sorted(stiffness_scales.items(), key=lambda x: int(x[0])):
        pct = (count / total_spurs) * 100 if total_spurs > 0 else 0
        print(f"Stiffness Scale {scale}: {count} ({pct:.1f}%)")


def read_comments(json_files):
    # --------------------------------------
    # STEP 3 — read comments from JSON files
    # --------------------------------------
    comments = []
    print('')
    for json_path in tqdm(json_files, desc="Reading comments from JSON files"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            comment = data.get("results", {}).get("comments")
            if comment != "N/A":
                comments.append((json_path, comment))

        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    print(f"Found {len(comments)} comments in JSON files")

    for json_path, comment in comments:
        print(f"{json_path}: {comment}")

    return comments




def main():

    # Path to your main directory
    BASE_DIR = "/media/alejo/IL_data/01_IL_bagfiles/experiment_1_(pull)"
    json_files_sd1 = collect_json_files(BASE_DIR)

    BASE_DIR_2 = "/media/alejo/New Volume/01_IL_bagfiles/experiment_4/"
    json_files_sd2 = collect_json_files(BASE_DIR_2)

    # Path to your main directory
    BASE_DIR_3 = "/media/alejo/IL_data/01_IL_bagfiles/only_human_demos"
    json_files_sd3 = collect_json_files(BASE_DIR_3)

    # Combine both lists
    json_files = json_files_sd1 + json_files_sd2 + json_files_sd3

    complete_json_file(json_files)
    count_apple_proxy_variations(json_files)
    # read_comments(json_files)



if __name__ == '__main__':
    main()
