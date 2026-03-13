import warnings
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tslearn.datasets import UCR_UEA_datasets

from src.processing import LSSTProcessor
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

# CHANGE 1: Point config_path to the root 'configs' folder
@hydra.main(version_base=None, config_path="src/configs", config_name="train_config")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    # Ensure src.utils.init_utils exists, otherwise use a simple logger
    try:
        logger = setup_saving_and_logging(config)
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
    
    writer = instantiate(config.writer)( logger=logger, project_config=project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # 1. Setup Dataset (Uses your dense collate_fn automatically)
    df = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = df.load_dataset("LSST")
    processor = LSSTProcessor()
    X_train, y_train = processor.fit_transform(X_train, y_train)
    X_test, y_test = processor.transform(X_test, y_test)
    
    model = instantiate(config.model)(device=device).from_pretrained(config.trainer.path)

    logger.info(f"Model: {config.model._target_}")

    # 3. Setup Loss and Metrics
    loss_function = instantiate(config.loss_function).to(device)

    # CHANGE 2: Manually instantiate metrics to ensure correct dict structure
    # This prevents errors if Hydra doesn't automatically parse the list structure
    # metrics = {
    #     "train": [instantiate(m) for m in config.metrics.train],
    #     "inference": [instantiate(m) for m in config.metrics.inference]
    # }

    # 4. Build Optimizer & Scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    init_optimizer = lambda params: instantiate(config.optimizer)(params=params)
    
    # lr_scheduler = None
    # if "lr_scheduler" in config and config.lr_scheduler is not None:
    #     lr_scheduler = instantiate(config.lr_scheduler)(optimizer=optimizer)

    # 5. Initialize Trainer
    trainer = instantiate(config.trainer)(
        network=model,
        device='cpu'
    )

    # 6. Run Training
    trainer.fit(X_train, y_train, 
                num_epochs=config.trainer.num_epochs, 
                fine_tuning_type=config.trainer.fine_tuning_type, 
                init_optimizer=init_optimizer
                )

if __name__ == "__main__":
    main()