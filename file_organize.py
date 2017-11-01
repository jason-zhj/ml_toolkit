"""
Utilities for organizing files
"""
import os, random
from shutil import copyfile


def generate_label_folders(source_dir="mnist_m_test", target_dir="test", mapping_file="mnist_m_test_labels.txt"):
    "this will move files from `source_dir` to subdirectories in `target_dir` according to `mapping_file`"
    lines = open(mapping_file).readlines()
    num_lines = len(lines)

    for i, line in enumerate(lines):
        filename, label = line.strip().split()
        # check whether dir is created
        move_to = "{}/{}".format(target_dir, label)
        if (not os.path.exists(move_to)):
            os.makedirs(move_to)
        # move the file to the dir
        copyfile("{}/{}".format(source_dir, filename), "{}/{}".format(move_to, filename))

        if (i != 0 and i % int(num_lines / 10) == 0):
            percent = 10 * i / int(num_lines / 10)
            print("finished {}%".format(percent))


def create_mini_dataset(source_dir, target_dir, sample_rate):
    "this move a portion (`sample_rate`) of files from `source_dir` to `target_dir`"
    labels = os.listdir(source_dir)
    for label in labels:
        subdir = os.path.join(source_dir, label)
        files = os.listdir(subdir)
        files = random.sample(files, int(sample_rate * len(files)))
        # move to target dir
        tgt_subdir = os.path.join(target_dir, label)
        if (not os.path.exists(tgt_subdir)): os.makedirs(tgt_subdir)
        for file in files:
            source_file = os.path.join(subdir, file)
            tgt_file = os.path.join(tgt_subdir, file)
            copyfile(source_file, tgt_file)

