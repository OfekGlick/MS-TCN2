import os
from tqdm import tqdm
from eval import f_score, edit_score, custom_get_labels_start_end_time, get_labels_start_end_time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gestures = {f"G{i}": i for i in range(6)}


def convert_file_to_list(gt_data):
    ground_truth = np.zeros(int(gt_data[-1].split()[1]))
    for row in gt_data:
        row = row.split()
        ground_truth[int(row[0]):int(row[1]) + 1] = gestures[row[2]]
    return ground_truth


def plot_segments(pred, gt):
    colors = ['red', 'blue', 'green', 'pink', 'orange', 'black']

    y_label, y_start, y_end = custom_get_labels_start_end_time(gt)
    df_1 = pd.DataFrame({'start_time': y_start, 'end_time': y_end, 'label': y_label})
    df_1.end_time = df_1.end_time - 1

    p_label, p_start, p_end = get_labels_start_end_time(pred)
    df_2 = pd.DataFrame({'start_time': p_start, 'end_time': p_end, 'label': p_label})
    df_2.end_time = df_2.end_time - 1

    for i in range(len(y_label)):
        plt.plot([df_1['start_time'][i], df_1['end_time'][i]], [0, 0], color=colors[gestures[df_1['label'][i]]],
                 linewidth=4)

    for i in range(len(p_label)):
        plt.plot([df_2['start_time'][i], df_2['end_time'][i]], [2, 2], color=colors[df_2['label'][i]], linewidth=4)
    plt.ylim(-2, 10)
    plt.show()


def accuracy(pred, gt):
    length = min(len(pred), len(gt))
    pred = np.array(pred[:length].copy())
    gt = gt[:length].copy()
    return (pred == gt).sum() / length


df = {"fold": [],
      "sample_size": [],
      "edit_score_average": [],
      "mean_accuracy": [],
      "f1@10": [],
      "f1@25": [],
      "f1@50": []
      }

pred_path = f"/home/student/FinalProject/Ofek/results/exp46"
gt_path = f"/datashare/APAS/transcriptions_gestures/"
folds = os.listdir(pred_path)
overlap = [.1, .25, .50]
for fold in sorted(folds):
    samples = os.listdir(os.path.join(pred_path, fold))
    print(f"Fold: {fold}")
    for sample in sorted(samples, key=lambda x: int(x.split()[-1])):
        print(f"\tTest: {sample}")
        preds = os.listdir(os.path.join(pred_path, fold, sample))
        edit_avg = []
        f1s = [0, 0, 0]
        acc = []
        for file in tqdm(preds):
            with open(os.path.join(pred_path, fold, sample, file), 'r') as pred_file:
                with open(os.path.join(gt_path, file + ".txt")) as gt_file:
                    pred_data = [gestures[row] for row in pred_file.readlines()[1].split()]
                    gt_data = [row.strip() for row in gt_file.readlines()]
                    gt_data_np = convert_file_to_list(gt_data)

                    acc.append(accuracy(pred_data, gt_data_np))

                    # plot_segments(pred_data, gt_data)
                    edit_avg.append(edit_score(pred_data, gt_data, test=True))
                    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
                    for s in range(len(overlap)):
                        tp1, fp1, fn1 = f_score(pred_data, gt_data, overlap[s], train=False)
                        tp[s] += tp1
                        fp[s] += fp1
                        fn[s] += fn1
                    for s in range(len(overlap)):
                        precision = tp[s] / float(tp[s] + fp[s])
                        recall = tp[s] / float(tp[s] + fn[s])
                        f1 = 2.0 * (precision * recall) / (precision + recall)
                        f1 = np.nan_to_num(f1) * 100
                        f1s[s] += f1
        df['fold'].append(fold)
        df['sample_size'].append(sample.split()[-1])
        df['edit_score_average'].append(np.mean(edit_avg))
        df['mean_accuracy'].append(np.mean(acc) * 100)
        print(f"\t\tEdit score average: {np.mean(edit_avg)}")
        print(f"\t\tEdit score Max: {np.max(edit_avg)}")
        print(f"\t\tEdit score Min: {np.min(edit_avg)}")
        print(f"\t\tMean Accuracy: {np.mean(acc)}")
        for i, k in enumerate(overlap):
            print(f"\t\tf1@{int(k * 100)}: {f1s[i] / len(preds)}")
            df[f"f1@{int(k * 100)}"].append(f1s[i] / len(preds))

df = pd.DataFrame.from_dict(df)
temp = df.groupby("sample_size").mean().reset_index()
temp['sample_size'] = temp['sample_size'].astype(int)
temp.to_csv("basekune.csv", index=False)

temp.sort_values(by='sample_size')
plt.plot(temp['sample_size'], temp['edit_score_average'], marker='o', label='Edit score average')
plt.plot(temp['sample_size'], temp['mean_accuracy'], label='Accuracy average', marker='o')
plt.plot(temp['sample_size'], temp['f1@10'], label='Mean F1@10', marker='o')
plt.plot(temp['sample_size'], temp['f1@25'], label='Mean F1@25', marker='o')
plt.plot(temp['sample_size'], temp['f1@50'], label='Mean F1@50', marker='o')
plt.xlabel("Sample size")
plt.xticks([1, 5, 10, 30, 60])
plt.legend()
plt.show()
ofek = 5
