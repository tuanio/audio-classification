import hydra
import argparse
from omegaconf import OmegaConf, DictConfig
from datamodule import SpeechCommandDataModule
from model import Model
import pytorch_lightning as pl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config path")
    parser.add_argument("-cp", help="config path")  # config path
    parser.add_argument("-cn", help="config name")  # config name

    args = parser.parse_args()

    @hydra.main(config_path=args.cp, config_name=args.cn)
    def main(cfg: DictConfig):
        dm = SpeechCommandDataModule(**cfg.datamodule)
        model = Model(**cfg.model)

        logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.logger)

        trainer = pl.Trainer(logger=logger, **cfg.trainer)

        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

    main()
