import os
import platform
import json
from collections import Counter

import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import numpy as np


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

    proxy_json_path = Path(__file__).parent / 'data' / 'apple_proxy.json'

    # Open reference apple_proxy.json to get all possible variations
    with open(proxy_json_path, "r") as ap_prox:
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


def count_results(json_files):

    proxy_json_path = Path(__file__).parent / 'data' / 'apple_proxy.json'

    # Open reference apple_proxy.json to get all possible variations
    with open(proxy_json_path, "r") as ap_prox:
        reference_data = json.load(ap_prox)
    spurs_ref = reference_data.get("spurs", {})
    apples_ref = reference_data.get("apples", {})

    approach_success_counter = Counter()
    grasp_success_counter = Counter()
    pick_success_counter = Counter()
    disposal_success_counter = Counter()

    total_trials = 0

    for json_path in tqdm(json_files, desc="Processing trials"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            approach = data.get("results").get("success_approach")
            grasp = data.get("results").get("success_grasp")
            pick = data.get("results").get("success_pick")
            disposal = data.get("results").get("success_disposal")

            approach_success_counter[approach] += 1 
            grasp_success_counter[grasp] += 1 
            pick_success_counter[pick] += 1 
            disposal_success_counter[disposal] += 1 

            total_trials += 1

        except Exception as e:
            print(f"Error reading {json_path}: {e}")


    print("\n=== Trials Counts ===: ", total_trials)
    for approach, count in sorted(approach_success_counter.items(), key=lambda x: int(x[0])):
        pct = (count / total_trials) * 100 if total_trials > 0 else 0
        print(f"Approach {approach}: {count} ({pct:.1f}%)")
    print("\n====================")
    for grasp, count in sorted(grasp_success_counter.items(), key=lambda x: x[0]):
        pct = (count / total_trials) * 100 if total_trials > 0 else 0
        print(f"Grasp {grasp}: {count} ({pct:.1f}%)")   
    print("\n====================")
    for pick, count in sorted(pick_success_counter.items(), key=lambda x: x[0]):
        pct = (count / total_trials) * 100 if total_trials > 0 else 0
        print(f"Pick {pick}: {count} ({pct:.1f}%)")
    print("\n====================")
    for disposal, count in sorted(disposal_success_counter.items(), key=lambda x: x[0]):
        pct = (count / total_trials) * 100 if total_trials > 0 else 0
        print(f"Disposal {disposal}: {count} ({pct:.1f}%)")


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


def demonstrations_metadata():

    if platform.system() == "Windows":

        BASE_DIR_1 = os.path.join(r'D:',
                                   '01_IL_bagfiles',
                                   'experiment_1_(pull)')
        BASE_DIR_2 = os.path.join(r'E:'
                                   '01_IL_bagfiles',
                                   'experiment_1_(pull)')
        BASE_DIR_3 = os.path.join(r'D:',
                                   '01_IL_bagfiles',
                                   'only_human_demos',
                                   'with_palm_cam')

    else:

        BASE_DIR_1 = "/media/alejo/IL_data/01_IL_bagfiles/experiment_1_(pull)"
        BASE_DIR_2 = "/media/alejo/New Volume/01_IL_bagfiles/experiment_4/"
        BASE_DIR_3 = "/media/alejo/IL_data/01_IL_bagfiles/only_human_demos"

    # Combine lists
    json_files_sd1 = collect_json_files(BASE_DIR_1)
    json_files_sd2 = collect_json_files(BASE_DIR_2)
    json_files_sd3 = collect_json_files(BASE_DIR_3)

    json_files = json_files_sd1 + json_files_sd2 + json_files_sd3

    # # complete_json_file(json_files)
    count_apple_proxy_variations(json_files)
    # # read_comments(json_files)
    # count_results(json_files)


def implementation_metadata():

    if platform.system() == "Windows":
        APPROACH_DIR = os.path.join(r'D:',
                                     'DATA',
                                     '07_IL_implementation',
                                     'bagfiles',
                                     'experiment_1_(pull)',
                                     'approach')
    else:
        pass

    approach_json_files = collect_json_files(APPROACH_DIR)

    two_inputs_w_servo = []
    two_inputs_wo_servo = []
    three_inputs_w_servo = []
    three_inputs_wo_servo = []
    two_inputs_w_servo_xy = []
    two_inputs_wo_servo_xy = []
    three_inputs_w_servo_xy = []
    three_inputs_wo_servo_xy = []

    two_inputs_w_servo_ctr = 0.0
    two_inputs_wo_servo_ctr = 0.0
    three_inputs_w_servo_ctr = 0.0
    three_inputs_wo_servo_ctr = 0.0
    two_inputs_w_servo_xy_ctr = 0.0
    two_inputs_wo_servo_xy_ctr = 0.0
    three_inputs_w_servo_xy_ctr = 0.0
    three_inputs_wo_servo_xy_ctr = 0.0

    for approach_json_path in approach_json_files:

        with open(approach_json_path, "r") as f:
            data = json.load(f)

        # Controllers info
        controllers = data.get("controllers", {})
        approach_controller = controllers.get("approach", {})
        pi_gain_controller = approach_controller.get("PI", {}).get("PI gain", {})
        states = approach_controller.get("states", {})

        # Proxy info
        proxy = data.get("proxy", {})
        apple = proxy.get("apple", {})
        spur = proxy.get("spur", {})
        apple_id = apple.get("id")

        # Results info
        results = data.get("results", {})
        final_pose = results.get("approach metrics", {}).get("final pose", {})
        approach_res = results.get("approach metrics", {}).get("success", {})
        approach_sing = results.get("singularity", {})
        comments = results.get('comments',{})

        xy_error = np.linalg.norm([final_pose[0], final_pose[1]]) * 1000
        net_error = np.linalg.norm(final_pose) * 1000

        if states == ["tof", "inhand_cam_features"]:
            # --- TOF + Latent Vector ----
            if pi_gain_controller == 0.0:
                # --- Without Visual servo ---
                two_inputs_wo_servo.append(net_error)
                two_inputs_wo_servo_xy.append(xy_error)


            else:
                # --- With Visual servo ----
                two_inputs_w_servo.append(net_error)
                two_inputs_w_servo_xy.append(xy_error)


        elif states == ["tof", "inhand_cam_features", "apple_prior"]:
            # --- TOF + Latent Vector + Prior ----
            if pi_gain_controller == 0.0:
                # --- Without Visual servo ---
                three_inputs_wo_servo.append(net_error)
                three_inputs_wo_servo_xy.append(xy_error)

            else:
                # --- With Visual servo ----
                three_inputs_w_servo.append(net_error)
                three_inputs_w_servo_xy.append(xy_error)

        if approach_res and comments != 'N/A': print(approach_json_path)

    # =============== NET APPROACH ERROR ===============
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

    labels = [
        "2 inputs\nIL twist + v.servo (x,y)",
        "2 inputs\nIL twist",
        "2 inputs + prior apple pose\nIL twist + v.servo (x,y)",
        "2 inputs + prior apple pose\nIL twist",
    ]

    axes[0].boxplot(two_inputs_w_servo)
    axes[0].set_xticks([1])
    axes[0].set_xticklabels([labels[0]])

    axes[1].boxplot(two_inputs_wo_servo)
    axes[1].set_xticks([1])
    axes[1].set_xticklabels([labels[1]])

    axes[2].boxplot(three_inputs_w_servo)
    axes[2].set_xticks([1])
    axes[2].set_xticklabels([labels[2]])

    axes[3].boxplot(three_inputs_wo_servo)
    axes[3].set_xticks([1])
    axes[3].set_xticklabels([labels[3]])
    for ax in axes:
        ax.grid(True)

    fig.supylabel("Nearest net pose [mm]")
    plt.tight_layout()

    # =============== NET APPROACH ERROR ===============
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

    labels = [
        "2 inputs\nIL twist + v.servo (x,y)",
        "2 inputs\nIL twist",
        "2 inputs + prior apple pose\nIL twist + v.servo (x,y)",
        "2 inputs + prior apple pose\nIL twist",
    ]

    axes[0].boxplot(two_inputs_w_servo_xy)
    axes[0].set_xticks([1])
    axes[0].set_xticklabels([labels[0]])

    axes[1].boxplot(two_inputs_wo_servo_xy)
    axes[1].set_xticks([1])
    axes[1].set_xticklabels([labels[1]])

    axes[2].boxplot(three_inputs_w_servo_xy)
    axes[2].set_xticks([1])
    axes[2].set_xticklabels([labels[2]])

    axes[3].boxplot(three_inputs_wo_servo_xy)
    axes[3].set_xticks([1])
    axes[3].set_xticklabels([labels[3]])
    for ax in axes:
        ax.grid(True)

    fig.supylabel("Nearest XY pose [mm]")
    plt.tight_layout()

    plt.show()

    pass


if __name__ == '__main__':
    # demonstrations_metadata()

    implementation_metadata()