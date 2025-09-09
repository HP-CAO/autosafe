import os
import json
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import tqdm

root_path = "."  # run under the results folder
all_data_path = "data_source"
case_study_list = ["cartpole", "glucose", "quadrotor", "quadruped"]
algorithm_list = ["model_based", "sac_base", "sac_autosafe_exp", "sac_autosafe_linear",
                  "sac_autosafe_opt", "sac_lag", "sac_lam_exp", "sac_lam_linear",
                  "sac_lam_opt", "sac_lyapunov", "sac_residual", "sac_simplex"]

data_info = {}
for case in case_study_list:
    data_info[case] = {}
    for algo in algorithm_list:
        data_info[case][algo] = []
data_info_json_path = os.path.join(root_path, "data_info.json")

def parse_event_to_csv(event_path, output_csv):
    event_acc = event_accumulator.EventAccumulator(event_path)
    event_acc.Reload()

    # Extract scalars
    tags = event_acc.Tags()["scalars"]
    data_dict = {tag: [] for tag in tags}

    for tag in tags:
        events = event_acc.Scalars(tag)
        for event in events:
            data_dict[tag].append([event.step, event.wall_time, event.value])

    # Convert to DataFrame
    all_dfs = []
    for tag, values in data_dict.items():
        df = pd.DataFrame(values, columns=["step", "wall_time", "value"])
        df["tag"] = tag
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Save to CSV
    final_df.to_csv(output_csv, index=False)


for case in case_study_list:
    os.makedirs(os.path.join(root_path, case), exist_ok=True)
    for algorithm in algorithm_list:
        os.makedirs(os.path.join(root_path, case, algorithm),exist_ok=True)
all_file = os.listdir(all_data_path)
for file_name in tqdm.tqdm(all_file):
    data_path = os.path.join(all_data_path, file_name)
    for i in os.listdir(data_path):
        case_study_name = file_name
        algo_name = i
        for seed in os.listdir(os.path.join(data_path, i )):
            tensorboard_event_folder = os.path.join(data_path, i, seed)
            tensorboard_event_name = os.listdir(tensorboard_event_folder)
            tensorboard_event_name.sort()
            tensorboard_event_name = tensorboard_event_name[0]
            tensorboard_event_path = os.path.join(tensorboard_event_folder, tensorboard_event_name)
            save_name = seed# run number

            print(f"Processing {case_study_name} {algo_name} {save_name}")
            data_info[case_study_name][algo_name].append(save_name)
            save_path = os.path.join(root_path, case_study_name, algo_name, save_name + ".csv")
            print(tensorboard_event_path)
            cvs_file = parse_event_to_csv(tensorboard_event_path, save_path)

print(f"Data parsing completed successfully, all data are saved to {data_info_json_path}")
f = open(data_info_json_path, "w")
json_data = json.dumps(data_info, default=lambda o: o.__dict__, indent=4)
f.write(json_data)
f.close()
