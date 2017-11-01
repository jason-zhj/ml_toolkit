# the training script which contains a function: def training(params)
# the function should return a dict whose content will be written to a result file
training_module = "training"
training_module_dir = "F:/Project/temp-projects/pytorch-test"

# `search_method` should be either "loopall" or "besteach"
# "loopall" will try all different combinations of parameter values
# "besteach" will keep all other parameters fixed as the default when testing with each parameter
search_method = "besteach"

# hyper parameters to test with
search_params = dict(
    learning_rate = [0.001,0.01,0.05],
    hash_size = [8,16,32],
    iterations = [100,200,300]
)

# default settings
default_param = dict(
    learning_rate = 0.01,
    # parameters for image preprocessing
    dataset_mean = (0.5, 0.5, 0.5),
    dataset_std = (0.5, 0.5, 0.5),

    # parameters for the model
    hash_size = 8,

    # settings for training
    train_shared_feat = False,
    train_specific_hash = True,
    final_feat_loss = "hash", # classification or hash
    image_scale = 100,
    shuffle_batch = True,

    # parameters for training
    source_data_path = "F:/data/domain_adaptation_images/train-test-split-for-domadv/mini/train/source",
    target_data_path = "F:/data/domain_adaptation_images/train-test-split-for-domadv/mini/train/target",
    iterations = 100,
    batch_size = 4
)



# path to save result file
save_result_to = "result.csv"