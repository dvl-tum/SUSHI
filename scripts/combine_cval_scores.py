import pandas as pd
from configs.config import get_arguments
from src.utils.deterministic import make_deterministic
import os.path as osp
import matplotlib.pyplot as plt
import os
import shutil

# Setup
config = get_arguments()
make_deterministic(config.seed)
# Search for directories containing the run_string, followed by '_split'
exp_path = osp.join(config.output_path, 'experiments')
if config.run_id is not None:
    exp_db = os.listdir(exp_path)
    exp_paths = [osp.join(exp_path,exp_dir)  for exp_dir in exp_db if exp_dir.startswith(config.run_id.strip() + '_split')]
    assert len(exp_paths) == 3, f"We found more than 3 runs. Comment this out if running on MOT20! {exp_paths}"
    print("GETTING PATHS FROM ", exp_paths)

else:
    exp_paths = [osp.join(config.output_path, 'experiments', exp_id) for exp_id in config.old_experiments]

# Determine the epoch range for which we have results for all runs 
max_epoch = min(max([int(dir.split('Epoch')[-1]) for dir in os.listdir(exp_path) if dir.startswith('Epoch')]) for exp_path in exp_paths)
data_file = 'mot_files/pedestrian_detailed.csv'
start_epoch = max(min([int(dir.split('Epoch')[-1]) for dir in os.listdir(exp_path) if dir.startswith('Epoch') and osp.exists(osp.join(exp_path, dir, data_file))]) for exp_path in exp_paths)
config.num_epoch = min(config.num_epoch, int(max_epoch))

# Results will be stored in a dict
scores = {"MOTA": [], "IDF1": [], "Total": []}

# Loop over the epochs
for e in range(start_epoch, config.num_epoch+1):
    epoch_logs = {"CLR_FN": 0, "CLR_FP": 0, "CLR_TP":0, "IDSW": 0, "IDTP": 0, "IDFN":0, "IDFP":0}
    epoch_dir = 'Epoch'+str(e)
    # Loop over the experiments
    for exp_path in exp_paths:
        # Get the csv files for each experiment
        mot_path = osp.join(exp_path, epoch_dir, config.mot_sub_folder, 'pedestrian_detailed.csv')
        mot_csv = pd.read_csv(mot_path)
        overall_csv = mot_csv[mot_csv['seq'] == 'COMBINED']

        # Combine the primitive metrics
        epoch_logs["CLR_FN"] += overall_csv["CLR_FN"].values[0]
        epoch_logs["CLR_FP"] += overall_csv["CLR_FP"].values[0]
        epoch_logs["CLR_TP"] += overall_csv["CLR_TP"].values[0]
        epoch_logs["IDSW"] += overall_csv["IDSW"].values[0]
        epoch_logs["IDTP"] += overall_csv["IDTP"].values[0]
        epoch_logs["IDFN"] += overall_csv["IDFN"].values[0]
        epoch_logs["IDFP"] += overall_csv["IDFP"].values[0]
    
    # Calculate MOTA and IDF1
    mota = (1 - (epoch_logs["CLR_FN"] + epoch_logs["CLR_FP"] + epoch_logs["IDSW"]) / (epoch_logs["CLR_TP"] + epoch_logs["CLR_FN"])) * 100
    idf1 = (epoch_logs["IDTP"]/(epoch_logs["IDTP"]+0.5*epoch_logs["IDFN"]+0.5*epoch_logs["IDFP"])) * 100
    total = (mota+idf1)/2
    mota = round(mota, 2)
    idf1 = round(idf1, 2)
    total = round(total, 2)
    
    scores["MOTA"].append(mota)
    scores["IDF1"].append(idf1)
    scores["Total"].append(total)  # Will be used to get the best overall performance epoch


# Moving average scores
window_length = 1
average_scores = {"MOTA": [], "IDF1": [], "Total": []}

for start_ix in range(0, config.num_epoch - window_length + 2 - start_epoch):
    end_ix = start_ix + window_length
    mota = sum(scores["MOTA"][start_ix:end_ix]) / window_length
    idf1 = sum(scores["IDF1"][start_ix:end_ix]) / window_length
    total = sum(scores["Total"][start_ix:end_ix]) / window_length
    mota = round(mota, 2)
    idf1 = round(idf1, 2)
    total = round(total, 2)

    average_scores["MOTA"].append(mota)
    average_scores["IDF1"].append(idf1)
    average_scores["Total"].append(total)


    print(f"Evaluating Epochs: {start_ix + 1 + start_epoch}-{end_ix + start_epoch}")
    print(f" MOTA: {mota}     IDF1: {idf1}      Total:{total}")

# Print out the best scores
print("----------------------------------------")
print("BEST RESULTS")
print(f"mMOTA: {max(average_scores['MOTA'])} | Epoch {average_scores['MOTA'].index(max(average_scores['MOTA']))+1}-{average_scores['MOTA'].index(max(average_scores['MOTA']))+window_length}")
print(f"MOTA: {max(scores['MOTA'])} | Epoch {scores['MOTA'].index(max(scores['MOTA']))+1}")
print(f"mIDF1: {max(average_scores['IDF1'])} | Epoch {average_scores['IDF1'].index(max(average_scores['IDF1']))+1}-{average_scores['IDF1'].index(max(average_scores['IDF1']))+window_length}")
print(f"IDF1: {max(scores['IDF1'])} | Epoch {scores['IDF1'].index(max(scores['IDF1']))+1}")
mtotal_ix = average_scores['Total'].index(max(average_scores['Total']))
total_ix = scores['Total'].index(max(scores['Total']))
print(f"mTotal: {average_scores['MOTA'][mtotal_ix]} / {average_scores['IDF1'][mtotal_ix]} | Epoch {mtotal_ix+1}-{mtotal_ix+window_length}")
print(f"Total: {scores['MOTA'][total_ix]} / {scores['IDF1'][total_ix]} | Epoch {total_ix+1}")

print(f"mTotal: MOTA {average_scores['MOTA'][mtotal_ix]} / IDF1 {average_scores['IDF1'][mtotal_ix]}| Epoch {mtotal_ix+1}-{mtotal_ix+window_length}")

# Plot the scores
x_scores = [i for i in range(1, len(scores['MOTA'])+1)]
s = 0  # Iteration to start vis from
e = 150  # Iteration to end the vis
plt.plot(x_scores[s:e], scores['MOTA'][s:e], label='MOTA')
plt.plot(x_scores[s:e], scores['IDF1'][s:e], label='IDF1')
plt.plot(x_scores[s:e], scores['Total'][s:e], label='Total')
plt.legend()
plt.show()
plt.close()

x_average_scores = [i for i in range(1, len(average_scores['MOTA'])+1)]
plt.plot(x_average_scores[s:e], average_scores['MOTA'][s:e], label='mMOTA')
plt.plot(x_average_scores[s:e], average_scores['IDF1'][s:e], label='mIDF1')
plt.plot(x_average_scores[s:e], average_scores['Total'][s:e], label='mTotal')
plt.legend()
plt.show()
plt.close()



