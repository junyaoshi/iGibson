from matplotlib import pyplot as plt
import os
import numpy as np

# clip params
scene_id = "Rs_int"
use_colors = False
# note: need to use underscore for category, e.g. towel_rack
target_name = 'grey door' if use_colors else 'door'
target_data_dir = f'/home/junyaoshi/Desktop/CLIP_semantics_plots/{scene_id}_{target_name.replace(" ", "_")}'
plot_file = os.path.join(target_data_dir, f'{scene_id}_{target_name.replace(" ", "_")}_0831_1.png')
plt.figure(figsize=(30, 30))
target_x, target_y = np.load(os.path.join(target_data_dir, 'target_xy.npy'))
plt.plot(target_x, target_y, marker='*', markersize=100)
data_dirs = [f.path for f in os.scandir(target_data_dir) if f.is_dir()]
all_scores = []
for data_dir in data_dirs:
    if os.listdir(data_dir):
        xs = np.load(os.path.join(data_dir, 'xs.npy'))
        ys = np.load(os.path.join(data_dir, 'ys.npy'))
        scores = np.load(os.path.join(data_dir, 'scores.npy'))
        plt.scatter(xs, ys, s=scores * 10000., alpha=0.7, c='r')
        all_scores.extend(list(scores))
plt.grid(True)
plt.title(f'{target_name} heat map in {scene_id}', fontsize=40)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=24)
all_scores = np.array(all_scores)
msizes = [all_scores.max() / 10 * 10000, all_scores.max() / 4 * 10000, all_scores.max() * 10000]
labels = [round(all_scores.max() / 10, 2),
          round(all_scores.max() / 4, 2),
          round(all_scores.max(), 2)]
markers = []
for size in msizes:
   markers.append(plt.scatter([],[], s=size, label=size, c='r'))
plt.legend(handles=markers, labels=labels, fontsize=40)
plt.tight_layout()
plt.savefig(plot_file)
plt.close()