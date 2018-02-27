"""
Utilities for organizing files
"""
import os, random
from shutil import copyfile

def _block_copy(source_dir,filenames,target_dir):
    "copy `filenames` from source_dir to target_dir"
    for file in filenames:
        source_file = os.path.join(source_dir, file)
        tgt_file = os.path.join(target_dir, file)
        copyfile(source_file, tgt_file)

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


def create_mini_dataset(source_dir, target_dir, sample_rate, except_dir=None):
    "this move a portion (`sample_rate`) of files from `source_dir` to `target_dir`"
    labels = os.listdir(source_dir)
    for label in labels:
        subdir = os.path.join(source_dir, label)
        files = os.listdir(subdir)
        num_files = len(files) # total number of files
        if (except_dir is not None):
            except_subdir = os.path.join(except_dir,label)
            except_files = os.listdir(except_subdir)
            files = [f for f in files if f not in except_files]

        files = random.sample(files, int(sample_rate * num_files))
        # move to target dir
        tgt_subdir = os.path.join(target_dir, label)
        if (not os.path.exists(tgt_subdir)): os.makedirs(tgt_subdir)
        for file in files:
            source_file = os.path.join(subdir, file)
            tgt_file = os.path.join(tgt_subdir, file)
            copyfile(source_file, tgt_file)

def create_dataset_split(source_dir,target_dirs,split_ratio):
    """
    :param source_dir: this folder should contain class labels as sub-directoriy names
    :param target_dirs: a list of two dir names, where the files from `source_dir` will be copied in
    :param split_ratio: 0~1
    """
    target_A, target_B = target_dirs
    labels = os.listdir(source_dir)
    for label in labels:
        subdir = os.path.join(source_dir, label)
        files = os.listdir(subdir)
        files_to_A = random.sample(files, int(split_ratio * len(files)))
        files_to_B = [f for f in files if f not in files_to_A]
        # move to `files_to_A` to target_A
        tgt_A_subdir = os.path.join(target_A, label)
        if (not os.path.exists(tgt_A_subdir)): os.makedirs(tgt_A_subdir)
        _block_copy(source_dir=subdir,target_dir=tgt_A_subdir,filenames=files_to_A)
        # move to `files_to_B` to target_B
        tgt_B_subdir = os.path.join(target_B, label)
        if (not os.path.exists(tgt_B_subdir)): os.makedirs(tgt_B_subdir)
        _block_copy(source_dir=subdir, target_dir=tgt_B_subdir, filenames=files_to_B)


def combine_dataset(source_dirs,target_dir):
    "copy files from `source_dirs` to `target_dir`"
    for source_dir in source_dirs:
        # copy files from source dir to target dir
        labels = os.listdir(source_dir)
        for label in labels:
            source_subdir = os.path.join(source_dir,label)
            target_subdir = os.path.join(target_dir,label)
            if (not os.path.exists(target_subdir)): os.makedirs(target_subdir)
            _block_copy(source_dir=source_subdir,target_dir=target_subdir,filenames=os.listdir(source_subdir))