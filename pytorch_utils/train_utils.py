import os
import torch
from ml_toolkit.log_analyze import plot_trend_graph

def save_models(models,save_model_to,save_obj=True,save_params=True):
    "models is a dict {name:model_obj}"
    for name,model in models.items():
        if (save_obj):
            torch.save(model, os.path.join(save_model_to, "{}.model".format(name)))
        if (save_params):
            torch.save(model.state_dict(), os.path.join(save_model_to, "{}.params".format(name)))


def save_loss_records(loss_records,save_to, loss_name):
    "save the loss records as a graph"
    lines = [str(l) for l in loss_records]
    plot_trend_graph(var_names=[loss_name], var_indexes=[-1], var_types=["float"], var_colors=["r"], lines=lines,
                     title=loss_name, save_to=os.path.join(save_to, "{}.png".format(loss_name)), show_fig=False)

"""
a singleton static logger to be shared by all scripts
"""
import logging

class LoggerGenerator():
    logger_dict = {}

    @staticmethod
    def get_logger(log_file_path):
        if (log_file_path not in LoggerGenerator.logger_dict.keys() ):
            print("Creating a logger that writes to {}".format(log_file_path))
            logger = logging.getLogger('myapp-{}'.format(log_file_path))
            hdlr = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr)
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
            LoggerGenerator.logger_dict[log_file_path] = logger
            return logger
        else:
            # logger already created
            return LoggerGenerator.logger_dict[log_file_path]

