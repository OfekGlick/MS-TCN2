#!/usr/bin/python2.7
from clearml import Task, Logger
import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
from os import listdir
from os.path import isfile, join
import numpy as np


def calc_class_weights(labels_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_labels_files = [f for f in listdir(labels_path) if isfile(join(labels_path, f))]
    possible_labels = [f"G{i}" for i in range(6)]
    class_counts = {k: 0 for k in possible_labels}
    sample_count = 0
    for label_file in class_labels_files:
        with open(os.path.join(labels_path, label_file), 'r') as file:
            content = [(line.split()[-1], int(line.split()[1]) - int(line.split()[0]) - 1) for line in file.readlines()]
            sample_count += sum([amount for _, amount in content])
            for class_label, amount in content:
                class_counts[class_label] += amount

    class_counts = {k: v / sample_count for k, v in class_counts.items()}
    median_freq = np.median(list(class_counts.values()))
    class_weights = sorted([(k, median_freq / v) for k, v in class_counts.items()], key=lambda x: x[0])
    return torch.tensor([weight for _, weight in class_weights]).float().to(device)


def fold_split(features_path, val_path, test_path):
    with open(val_path, 'r') as f:
        val_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]
    with open(test_path, 'r') as f:
        test_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]
    train_files = [f for f in listdir(features_path) if isfile(join(features_path, f))]
    train_files = list(set(train_files) - set(test_files + val_files))
    return train_files, val_files, test_files


def run_train(val_path_fold, test_path_fold, features_path_fold, device, results_dir, model_dir, actions_dict,
              num_layers_PG, num_layers_R, num_R,
              num_f_maps, num_epochs, bz, lr, features_dim, clogger, kl_flag=False,
              label_class_weights=None,
              gru_flag=False,
              weighted_flag=False,
              final_gru=True, sample_size=5):
    fold_num = features_path_fold.split("/")[-2]
    print(f"\t{fold_num}")
    gt_path = '/datashare/APAS/transcriptions_gestures/'
    class_weights = calc_class_weights(gt_path)
    sample_rate = 1
    num_classes = len(actions_dict)
    folder_class_labels_flag = "ClassWeighted" if label_class_weights else "NotClassWeighted"
    final_GRU_flag = "Final-GRU" if final_gru else "Final-NoGRU"
    sample_size_flag = f"Sample size {sample_size}"
    ################## Flags to print ######################
    exts = [final_GRU_flag, folder_class_labels_flag, sample_size_flag]
    ########################################################
    folder_name = results_dir.format(fold_num, "_".join(exts))
    model_folder_name = model_dir.format(fold_num, "_".join(exts))
    try:
        os.makedirs(folder_name)
        os.makedirs(model_folder_name)
    except:
        pass
    vid_list_file, vid_list_file_val, vid_list_file_test = fold_split(features_path_fold, val_path_fold,
                                                                      test_path_fold)
    batch_gen_train = BatchGenerator(num_classes, actions_dict, gt_path, features_path_fold,
                                     sample_rate)

    batch_gen_train.read_data(vid_list_file)
    batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, features_path_fold, sample_rate)

    batch_gen_val.read_data(vid_list_file_val)
    trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes,
                      fold_num, fold_num, weighted=weighted_flag, kl=kl_flag, gru=gru_flag,
                      class_weights=class_weights if label_class_weights else None, final_gru=final_gru,
                      sample_size=sample_size)
    trainer.train(model_folder_name, batch_gen_train, batch_gen_val, num_epochs=num_epochs, batch_size=bz,
                  learning_rate=lr, device=device, clogger=clogger)

    trainer.predict(model_folder_name, folder_name,
                    features_path_fold, vid_list_file_test, num_epochs, actions_dict, device,
                    sample_rate)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='train')
    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')

    parser.add_argument('--features_dim', default='1280', type=int)
    parser.add_argument('--bz', default='1', type=int)
    parser.add_argument('--lr', default='0.0005', type=float)

    parser.add_argument('--num_f_maps', default='65', type=int)

    # Need input
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_layers_PG', default=10, type=int)
    parser.add_argument('--num_layers_R', default=10, type=int)
    parser.add_argument('--num_R', default=3, type=int)

    args = parser.parse_args()

    num_epochs = args.num_epochs
    features_dim = args.features_dim
    bz = args.bz
    lr = args.lr

    num_layers_PG = args.num_layers_PG
    num_layers_R = args.num_layers_R
    num_R = args.num_R
    num_f_maps = args.num_f_maps

    fold_files = [(f"/datashare/APAS/folds/valid {i}.txt",
                   f"/datashare/APAS/folds/test {i}.txt",
                   f"/datashare/APAS/features/fold{i}/") for i in range(5)]

    mapping_file = "/datashare/APAS/mapping_gestures.txt"
    try:
        latest = max([int(exp.split("exp")[-1]) for exp in os.listdir("./results")])
    except Exception:
        latest = 0
    new_exp = f"exp{latest + 1}"
    print(f"Experiment: {new_exp}")
    results_dir = "./results/" + str(new_exp) + "/{}/{}"
    model_dir = "./models/" + str(new_exp) + "/{}/{}"

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    if args.action == "baseline":
        task = Task.init(project_name='CVSA - Final project', task_name=new_exp + " Baseline")
    elif args.action == "train":
        task = Task.init(project_name='CVSA - Final project', task_name=new_exp + " Chosen model")
    else:
        task = Task.init(project_name='CVSA - Final project', task_name=new_exp)

    clogger = task.get_logger()

    print("Starting training!")
    if args.action == "train_tradeoff":
        for val_path_fold, test_path_fold, features_path_fold in fold_files:
            for sample_size in [1, 5, 10, 30, 60]:
                run_train(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                          features_path_fold=features_path_fold, device=device, results_dir=results_dir,
                          model_dir=model_dir,
                          actions_dict=actions_dict,
                          num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                          num_f_maps=num_f_maps, num_epochs=num_epochs, bz=bz, lr=lr, features_dim=features_dim,
                          clogger=clogger, kl_flag=False,
                          label_class_weights=True,
                          gru_flag=False,
                          weighted_flag=False,
                          final_gru=True, sample_size=sample_size)

    if args.action == "train":
        for val_path_fold, test_path_fold, features_path_fold in fold_files:
            run_train(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                      features_path_fold=features_path_fold, device=device, results_dir=results_dir,
                      model_dir=model_dir,
                      actions_dict=actions_dict,
                      num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                      num_f_maps=num_f_maps, num_epochs=num_epochs, bz=bz, lr=lr, features_dim=features_dim,
                      clogger=clogger, kl_flag=False,
                      label_class_weights=True,
                      gru_flag=False,
                      weighted_flag=False,
                      final_gru=True, sample_size=5)

    if args.action == "baseline":
        for val_path_fold, test_path_fold, features_path_fold in fold_files:
            run_train(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                      features_path_fold=features_path_fold, device=device, results_dir=results_dir,
                      model_dir=model_dir,
                      actions_dict=actions_dict,
                      num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                      num_f_maps=num_f_maps, num_epochs=num_epochs, bz=bz, lr=lr, features_dim=features_dim,
                      clogger=clogger, kl_flag=False,
                      label_class_weights=None,
                      gru_flag=False,
                      weighted_flag=False,
                      final_gru=False, sample_size=1)


if __name__ == '__main__':
    main()
