import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

from base import BaseDataLoader

import os
from torch import sigmoid


def main(config):
    logger = config.get_logger("test")

    config["data_loader"]["args"]["is_train"] = False

    # setup data_loader instances
    dataset = config.init_obj("data_loader", module_data)
    data_loader = BaseDataLoader(
        dataset=dataset, batch_size=128, shuffle=False, validation_split_size=0.0, num_workers=0
    )

    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     # is_train=False,
    #     max_seq_len=config['data_loader']['args']['max_seq_len'],
    #     num_workers=0
    # )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))
    total_preds = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            outputs = sigmoid(outputs)
            outputs = outputs.cpu().detach().numpy()
            total_preds += list(outputs)

            # computing loss, metrics on test set
    #         loss = loss_fn(outputs, targets.float())
    #         batch_size = targets.shape[0]
    #         total_loss += loss.item() * batch_size
    #         for i, metric in enumerate(metric_fns):
    #             total_metrics[i] += metric(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy()) * batch_size

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)

    write_path = os.path.join("outputs/", "submission.csv")
    os.makedirs(name="outputs/", exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args)
    main(config)
