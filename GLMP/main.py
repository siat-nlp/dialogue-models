import os
import json
import logging
import argparse
import torch

from src.inputters.batcher import prepare_batcher, load_vocab
from src.inputters.dataset import Lang
from src.models.GLMP import GLMP
from src.utils.trainer import Trainer


def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data")
    data_arg.add_argument("--data_prefix", type=str, default="")
    data_arg.add_argument("--save_dir", type=str, default="./models")
    data_arg.add_argument("--output_dir", type=str, default="./output")

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--vocab_size", type=int, default=40000)
    net_arg.add_argument("--hidden_size", type=int, default=512)
    net_arg.add_argument("--embed_size", type=int, default=512)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--hop", type=int, default=1)
    net_arg.add_argument("--max_dec_len", type=int, default=25)
    net_arg.add_argument("--use_record", type=bool, default=True)
    net_arg.add_argument("--unk_mask", type=bool, default=True)

    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--batch_size", type=int, default=32)
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0001)
    train_arg.add_argument("--lr_decay", type=float, default=0.5)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.2)
    train_arg.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    train_arg.add_argument("--num_epochs", type=int, default=10)
    train_arg.add_argument("--gpu", type=int, default=0)
    train_arg.add_argument("--log_steps", type=int, default=100)
    train_arg.add_argument("--valid_steps", type=int, default=500)
    train_arg.add_argument("--ckpt", type=str)
    train_arg.add_argument("--check", action="store_true")
    train_arg.add_argument("--test", action="store_true")
    train_arg.add_argument("--interact", action="store_true")

    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = model_config()
    if config.check:
        config.save_dir = "./tmp/"
    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    device = config.gpu
    torch.cuda.set_device(device)

    # Data definition
    train_file = "%s/json.train.txt" % config.data_dir
    dev_file = "%s/json.dev.txt" % config.data_dir
    test_file = "%s/json.test.txt" % config.data_dir
    vocab_file = "%s/train.vocab" % config.data_dir

    word2index, index2word = load_vocab(vocab_file, config.vocab_size)
    vocab_size = len(word2index)

    train_iter = prepare_batcher(train_file, word2index, batch_size=config.batch_size, is_shuffle=True)
    valid_iter = prepare_batcher(dev_file, word2index, batch_size=config.batch_size, is_shuffle=True)
    test_iter = prepare_batcher(test_file, word2index, batch_size=config.batch_size, is_shuffle=False)

    # Model definition
    model = GLMP(index2word, vocab_size=vocab_size, hidden_size=config.hidden_size, embed_dim=config.embed_size,
                 max_resp_len=config.max_dec_len, n_layers=config.num_layers, hop=config.hop,
                 dropout=config.dropout, teacher_forcing_ratio=config.teacher_forcing_ratio,
                 use_cuda=config.use_gpu, use_record=config.use_record, unk_mask=config.unk_mask)

    # Testing
    if config.test and config.ckpt:
        print(model)
        model.load(save_dir=config.save_dir, file_prefix=config.ckpt)

        print("Generating ...")
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        model.generate(test_iter, output_dir=config.output_dir, verbose=True)
    else:
        # Save directory
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        # Optimizer definition
        optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.lr)
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=config.lr_decay, patience=1, verbose=True, min_lr=1e-5)

        # Logger definition
        logger = logging.getLogger()
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
        logger.addHandler(fh)
        # Save config
        params_file = os.path.join(config.save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Saved params to '{}'".format(params_file))
        logger.info(model)

        # Train
        logger.info("Training starts ...")
        trainer = Trainer(model=model, optimizer=optimizer, train_iter=train_iter,
                          valid_iter=valid_iter, logger=logger,
                          valid_metric_name="-loss", num_epochs=config.num_epochs,
                          save_dir=config.save_dir, log_steps=config.log_steps,
                          valid_steps=config.valid_steps, grad_clip=config.grad_clip,
                          lr_scheduler=lr_scheduler)
        if config.ckpt is not None:
            trainer.load(save_dir=config.save_dir, file_prefix=config.ckpt)
        trainer.train()
        logger.info("Training done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
