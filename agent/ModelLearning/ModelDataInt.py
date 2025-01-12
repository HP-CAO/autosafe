"""
This file defines the basic interaction pipeline between the model and the dataset.
"""

import logging
from tqdm import tqdm
import numpy as np


def train_loop(algo, tf_training_dataset, tf_val_dataset, writer, cfg):
    logging.info("Start Training Process")
    progress = tqdm(total=cfg.ModelParams.epochs * len(tf_training_dataset))
    best_performance = -100
    for ep in range(cfg.ModelParams.epochs):
        loss_list = []
        for iter, batch in enumerate(tf_training_dataset):
            loss = algo.optimize(batch)
            loss_list.append(loss)
            progress.set_description(f"Status: Epochs {ep}--Iters {iter}, loss={loss:.4f}")

        writer.add_scalar("losses/loss", np.mean(loss_list), ep)

        if ep % cfg.ModelParams.evaluation_period == 0:
            performance = val_loop(algo, tf_val_dataset)
            writer.add_scalar("eval/MSE", performance, ep)
            if performance > best_performance:
                best_performance = performance
                algo.save_weights(cfg.JobParams.output_path + '/tf_model/best_model/')

        progress.update(len(tf_training_dataset))
    algo.save_weights(cfg.JobParams.output_path + '/tf_model/final_model/')


def val_loop(algo, tf_eval_dataset):
    performance_list = []
    for batch in tf_eval_dataset:
        performance = algo.validate(batch)
        performance_list.append(performance)

    return np.mean(performance_list)


def eval_loop(model, tf_eval_dataset):
    raise NotImplementedError
