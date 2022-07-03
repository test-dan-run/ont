from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.datasets.lightning_speech_datasets import LightningSpeechDataset
from src.orchestrator import LightningONT

def training_pipeline(cfg: DictConfig):

    # setup dataset
    data_module = LightningSpeechDataset(cfg.data, cfg.run.batch_size)
    data_module.setup(stage='fit')

    # initialise orchestrator
    orchestrator = LightningONT(cfg.model, cfg.optim, cfg.run, cfg.data.n_fft, cfg.data.hop_length)

    # initialise callbacks
    checkpoint = ModelCheckpoint(monitor='valid_loss', filename='best', save_top_k=1)
    logger = TensorBoardLogger(cfg.run.logs_dir)

    # initialize Trainer
    trainer = Trainer(
        gpus=cfg.run.num_gpus, 
        callbacks=[checkpoint,], 
        logger=logger, 
        max_epochs=cfg.run.epochs, 
        strategy=DDPPlugin(find_unused_parameters=False) if cfg.run.num_gpus > 1 else None,
        )

    # train
    trainer.fit(orchestrator, datamodule=data_module)

    return trainer
    