import warnings
warnings.simplefilter('ignore', UserWarning)

import hydra
from omegaconf import DictConfig

@hydra.main(config_path='configs', config_name='local_train_base')
def train(cfg: DictConfig):
    from src.pipelines import training_pipeline
    
    training_pipeline(cfg)

if __name__ == '__main__':
    train()