"""
Automated Thesis Benchmarking Suite — FINAL CLEANED VERSION
Calculates ground-truth positional errors, records simulation task times,
and mathematically verifies task success without human intervention.
Metrics: Task_Time_Sec, Pick_Error_m, Drop_Error_m, Final_Distance_to_Bin_m, Success
"""

import os
import csv
import math
import random
from config import VLM_PATH, VLA_PATH, TARGET_NUDGES, TARGET_DEF,TARGET2_DEF, DEST_DEF, SIM_TABLE_DEF

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BIN_SUCCESS_RADIUS = 0.15

# =========================================================================
# 1. CORE LOGGING ENGINE (Baseline and OOD)
# =========================================================================

def _EVALUATE_AND_LOG(csv_filename, supervisor, architecture, start_time, end_time,
                    predicted_pick, predicted_drop, reference_target_pos,
                    target_def, bin_def, success_override=None):
    """
    Core math and logging engine for static tests.
    success_override: pass False to force-log a failure on abort or timeout.
    """
    csv_file = os.path.join(RESULTS_DIR, csv_filename)
    file_exists = os.path.isfile(csv_file)

    real_target_pos = supervisor.getFromDef(target_def).getPosition()
    real_bin_pos = supervisor.getFromDef(bin_def).getPosition()

    task_time = round(end_time - start_time, 3)
    pick_error = round(math.dist(predicted_pick, (reference_target_pos[0], reference_target_pos[1])), 4)
    drop_error = round(math.dist(predicted_drop, (real_bin_pos[0], real_bin_pos[1])), 4)
    final_distance_to_bin = round(math.dist(
        (real_target_pos[0], real_target_pos[1]),
        (real_bin_pos[0], real_bin_pos[1])
    ), 4)

    success = success_override if success_override is not None else bool(final_distance_to_bin < BIN_SUCCESS_RADIUS)
    model_id = VLM_PATH if architecture == "VLM" else VLA_PATH

    headers = ["Architecture", "Model_ID", "Task_Time_Sec", "Pick_Error_m",
               "Drop_Error_m", "Final_Distance_to_Bin_m", "Success"]
    row = [architecture, model_id, task_time, pick_error, drop_error, final_distance_to_bin, success]

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"[Logger] {csv_filename} | Success={success} | Time={task_time}s | PickErr={pick_error}m")


def LOG_BASELINE(supervisor, architecture, start_time, end_time,
                 predicted_pick, predicted_drop, initial_target_pos,
                 success_override=None):
    _EVALUATE_AND_LOG("baseline_metrics.csv", supervisor, architecture,
                      start_time, end_time, predicted_pick, predicted_drop,
                      initial_target_pos, target_def=TARGET_DEF, bin_def=DEST_DEF, success_override=success_override)


def LOG_OOD(supervisor, architecture, start_time, end_time,
            predicted_pick, predicted_drop, initial_target_pos,
            success_override=None):
    _EVALUATE_AND_LOG("ood_metrics.csv", supervisor, architecture,
                      start_time, end_time, predicted_pick, predicted_drop,
                      initial_target_pos, target_def=TARGET2_DEF, bin_def=DEST_DEF, success_override=success_override)


# =========================================================================
# 2. DYNAMIC NUDGE AUTOMATION AND LOGGING
# =========================================================================

def AUTOMATE_NUDGE(supervisor, target_def=TARGET_DEF, bin_def=DEST_DEF, table_def=SIM_TABLE_DEF):
    """
    Teleports the target object to a random position dynamically calculated 
    based on the table dimensions. Ensures the object stays on the table, 
    moves a minimum distance, and avoids colliding with the bin.
    """
    target_node = supervisor.getFromDef(target_def)
    bin_node = supervisor.getFromDef(bin_def)
    table_node = supervisor.getFromDef(table_def)

    target_pos = target_node.getPosition()
    bin_pos = bin_node.getPosition()
    table_pos = table_node.getPosition()
    table_size = table_node.getField("size").getSFVec3f()

    # Calculate safe manipulation zones (subtracting 6cm margin from edges)
    SAFETY_MARGIN = 0.06
    TABLE_X_MIN = table_pos[0] - (table_size[0] / 2) + SAFETY_MARGIN
    TABLE_X_MAX = table_pos[0] + (table_size[0] / 2) - SAFETY_MARGIN
    TABLE_Y_MIN = table_pos[1] - (table_size[1] / 2) + SAFETY_MARGIN
    TABLE_Y_MAX = table_pos[1] + (table_size[1] / 2) - SAFETY_MARGIN

    # Displacement parameters
    MIN_NUDGE_DIST = 0.08  # Ensure it moves at least 8cm so the robot must react
    MAX_NUDGE_DIST = 0.25  # Don't teleport it too far out of camera view
    BIN_AVOID_RADIUS = 0.20 # Keep it 20cm away from the bin's center

    # Try to find a valid safe position
    for _ in range(50):
        # Use polar coordinates to guarantee the minimum displacement distance
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(MIN_NUDGE_DIST, MAX_NUDGE_DIST)
        
        new_x = target_pos[0] + dist * math.cos(angle)
        new_y = target_pos[1] + dist * math.sin(angle)

        # 1. Check if it's falling off the table
        if not (TABLE_X_MIN < new_x < TABLE_X_MAX): continue
        if not (TABLE_Y_MIN < new_y < TABLE_Y_MAX): continue

        # 2. Check if it teleported into the bin
        dist_to_bin = math.dist((new_x, new_y), (bin_pos[0], bin_pos[1]))
        if dist_to_bin < BIN_AVOID_RADIUS: continue

        # If it passes both checks, apply the translation!
        new_x, new_y = round(new_x, 4), round(new_y, 4)
        target_node.getField("translation").setSFVec3f([new_x, new_y, target_pos[2]])

        # Settle physics
        timestep = int(supervisor.getBasicTimeStep())
        for _ in range(15):
            supervisor.step(timestep)

        return (new_x, new_y)

    # Fallback (Extremely rare: if 50 attempts fail, just bump it slightly)
    print("[Referee] Warning: Could not find optimal safe nudge spot. Forcing minor offset.")
    new_x, new_y = target_pos[0] + 0.1, target_pos[1]
    target_node.getField("translation").setSFVec3f([new_x, new_y, target_pos[2]])
    timestep = int(supervisor.getBasicTimeStep())
    for _ in range(15): supervisor.step(timestep)
    return (new_x, new_y)


def LOG_NUDGE_METRICS(supervisor, architecture, start_time, end_time,
                      nudge_history, final_pick_error, predicted_drop,
                      target_def=TARGET_DEF, bin_def=DEST_DEF):
    """
    Logger for the perturbation (nudge) test.
    nudge_history: list of dicts — [{'truth': (x,y), 'pred': (x,y), 'error': float}]
    """
    csv_file = os.path.join(RESULTS_DIR, "nudge_metrics.csv")
    file_exists = os.path.isfile(csv_file)

    real_bin_pos = supervisor.getFromDef(bin_def).getPosition()
    real_target_pos = supervisor.getFromDef(target_def).getPosition()

    task_time = round(end_time - start_time, 3)
    num_nudges = len(nudge_history)
    drop_error = round(math.dist(predicted_drop, (real_bin_pos[0], real_bin_pos[1])), 4) \
        if predicted_drop else 999.0
    final_distance_to_bin = round(math.dist(
        (real_target_pos[0], real_target_pos[1]),
        (real_bin_pos[0], real_bin_pos[1])
    ), 4)
    success = bool(final_distance_to_bin < BIN_SUCCESS_RADIUS)
    model_id = VLM_PATH if architecture == "VLM" else VLA_PATH

    headers = ["Architecture", "Model_ID", "Task_Time_Sec", "Total_Nudges",
               "Final_Pick_Error_m", "Drop_Error_m", "Final_Distance_to_Bin_m", "Success"]
    for i in range(1, TARGET_NUDGES + 1):
        headers.extend([f"Nudge_{i}_Truth", f"Nudge_{i}_Pred", f"Nudge_{i}_Error_m"])

    row = [architecture, model_id, task_time, num_nudges,
           round(final_pick_error, 4), drop_error, final_distance_to_bin, success]
    
    for i in range(TARGET_NUDGES):
        if i < len(nudge_history):
            n = nudge_history[i]
            row.extend([str(n['truth']), str(n['pred']), round(n['error'], 4)])
        else:
            row.extend(["N/A", "N/A", "N/A"])

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"[Logger] nudge_metrics.csv | Success={success} | Time={task_time}s | "
          f"Nudges={num_nudges} | PickErr={round(final_pick_error,4)}m")