"""
Automated Thesis Benchmarking Suite (The "Ghost Referee")
Calculates ground-truth positional errors, records simulation task times, 
and mathematically verifies task success without human intervention. 
"""

import os
import csv
import math
import random
from config import SELECTED_VLM, VLM_PATHS, VLA_PATH, TARGET_NUDGES, NUDGE_RANGE

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================================
# 1. CORE LOGGING ENGINE (For Baseline & OOD)
# =========================================================================

def _EVALUATE_AND_LOG(csv_filename, supervisor, architecture, start_time, end_time, predicted_pick, predicted_drop, reference_target_pos, target_def, bin_def):
    """
    Core math and logging engine for static tests. Handles the exact bounding-box 
    success verification and appends the trial data to the specified CSV file.
    """
    csv_file = os.path.join(RESULTS_DIR, csv_filename)
    file_exists = os.path.isfile(csv_file)

    real_target_node = supervisor.getFromDef(target_def)
    real_bin_node = supervisor.getFromDef(bin_def)
    
    real_target_pos = real_target_node.getPosition()
    real_bin_pos = real_bin_node.getPosition()

    # 1. Calculate Task Time
    task_time = round(end_time - start_time, 3)

    # 2. Calculate AI Prediction Precision 
    pick_error = round(math.dist(predicted_pick, (reference_target_pos[0], reference_target_pos[1])), 4)
    drop_error = round(math.dist(predicted_drop, (real_bin_pos[0], real_bin_pos[1])), 4)

    # 3. Calculate Physical Success (Did the object actually end up in the bin?)
    final_distance_to_bin = round(math.dist((real_target_pos[0], real_target_pos[1]), (real_bin_pos[0], real_bin_pos[1])), 4)
    success = bool(final_distance_to_bin < 0.15) # Assuming bin radius is 15cm

    # 4. Identify Model
    if architecture == "VLM":
        model_id = VLM_PATHS[SELECTED_VLM]['path']
    else:
        model_id = VLA_PATH

    # Write to CSV
    headers = ["Architecture", "Model_ID", "Task_Time_Sec", "Pick_Error_m", "Drop_Error_m", "Final_Distance_to_Bin_m", "Success"]
    row = [architecture, model_id, task_time, pick_error, drop_error, final_distance_to_bin, success]

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)


def LOG_BASELINE(supervisor, architecture, start_time, end_time, predicted_pick, predicted_drop, initial_target_pos, target_def="OBJ1", bin_def="BIN"):
    """Logs the standard static pick-and-place trial."""
    _EVALUATE_AND_LOG("baseline_metrics.csv", supervisor, architecture, start_time, end_time, predicted_pick, predicted_drop, initial_target_pos, target_def, bin_def)


def LOG_OOD(supervisor, architecture, start_time, end_time, predicted_pick, predicted_drop, initial_target_pos, target_def="OBJ2", bin_def="BIN"):
    """Logs the Out-Of-Distribution trial (e.g., the Jam Jar)."""
    _EVALUATE_AND_LOG("ood_metrics.csv", supervisor, architecture, start_time, end_time, predicted_pick, predicted_drop, initial_target_pos, target_def, bin_def)


# =========================================================================
# 2. DYNAMIC NUDGE TRACKING (For Reactive Tests)
# =========================================================================

def AUTOMATE_NUDGE(supervisor, target_def="OBJ1"):
    """
    Programmatically teleports the target to a random valid location 
    on the table to trigger a mid-task perturbation.
    """
    node = supervisor.getFromDef(target_def)
    pos = node.getPosition()
    
    dx = random.uniform(*NUDGE_RANGE) * random.choice([-1, 1])
    dy = random.uniform(*NUDGE_RANGE) * random.choice([-1, 1])
    
    new_x = round(pos[0] + dx, 3)
    new_y = round(pos[1] + dy, 3)
    
    # Teleport the object
    node.getField("translation").setSFVec3f([new_x, new_y, pos[2]])
    
    # Step physics to register the physical teleport
    timestep = int(supervisor.getBasicTimeStep())
    for _ in range(10): supervisor.step(timestep)
    
    return (new_x, new_y)


def LOG_NUDGE_METRICS(supervisor, architecture, start_time, end_time, nudge_history, final_pick_error, predicted_drop, target_def="OBJ1", bin_def="BIN"):
    """
    Dedicated logger for Dynamic Tracking.
    nudge_history is a list of dicts: [{'truth': (x,y), 'pred': (px,py), 'error': 0.04}, ...]
    """
    csv_file = os.path.join(RESULTS_DIR, "nudge_metrics.csv")
    file_exists = os.path.isfile(csv_file)

    task_time = round(end_time - start_time, 3)
    num_nudges = len(nudge_history)

    real_bin_node = supervisor.getFromDef(bin_def)
    real_bin_pos = real_bin_node.getPosition()
    
    # --- NEW: Grab the target's final physical position ---
    real_target_node = supervisor.getFromDef(target_def)
    real_target_pos = real_target_node.getPosition()
    # ------------------------------------------------------

    drop_error = round(math.dist(predicted_drop, (real_bin_pos[0], real_bin_pos[1])), 4) if predicted_drop else 999.0
    
    # --- NEW: Calculate physical success metrics ---
    final_distance_to_bin = round(math.dist((real_target_pos[0], real_target_pos[1]), (real_bin_pos[0], real_bin_pos[1])), 4)
    success = bool(final_distance_to_bin < 0.15) 
    # -----------------------------------------------

    if architecture == "VLM":
        model_id = VLM_PATHS[SELECTED_VLM]['path']
    else:
        model_id = VLA_PATH

    headers = ["Architecture", "Model_ID", "Task_Time_Sec", "Total_Nudges", "Final_Pick_Error_m", "Drop_Error_m", "Final_Distance_to_Bin_m", "Success"]
    
    for i in range(1, TARGET_NUDGES + 1):
        headers.extend([f"Nudge_{i}_Truth", f"Nudge_{i}_Pred", f"Nudge_{i}_Error_m"])
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        
        row = [architecture, model_id, task_time, num_nudges, round(final_pick_error, 4), drop_error, final_distance_to_bin, success]
        
        for i in range(TARGET_NUDGES):
            if i < len(nudge_history):
                n = nudge_history[i]
                row.extend([f"{n['truth']}", f"{n['pred']}", round(n['error'], 4)])
            else:
                row.extend(["N/A", "N/A", "N/A"]) 
                
        writer.writerow(row)