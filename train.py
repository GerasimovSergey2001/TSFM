import warnings
import os
from pathlib import Path
import json
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tslearn.datasets import UCR_UEA_datasets

from sklearn.metrics import classification_report

from src.processing import LSSTProcessor
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

# Point config_path to the root 'configs' folder
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

    ckpt_dir = Path(config.trainer.checkpoints)
    results_train_dir = Path(config.trainer.results + '/train')
    results_test_dir = Path(config.trainer.results + '/test')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_train_dir.mkdir(parents=True, exist_ok=True)
    results_test_dir.mkdir(parents=True, exist_ok=True)
    

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # Setup Dataset
    df = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = df.load_dataset("LSST")
    processor = LSSTProcessor()
    X_train, y_train = processor.fit_transform(X_train, y_train)
    X_test, y_test = processor.transform(X_test, y_test)
    
    model = instantiate(config.model)(device=device)

    logger.info(f"Model: {config.model._target_}")

    init_optimizer = None
    if config.optimizer is not None:
        init_optimizer = instantiate(config.optimizer)
    
    # Initialize Trainer
    trainer = instantiate(config.train_wrapper)(
        network=model.from_pretrained(config.trainer.path),
        device=device
    )

    trainer_config = config.trainer

    # Run Training
    trainer.fit(X_train, y_train, 
                num_epochs=trainer_config.num_epochs, 
                fine_tuning_type=trainer_config.fine_tuning_type, 
                init_optimizer=init_optimizer
                )
    
    checpkpoint_path = trainer_config.checkpoint_name+'_'+ trainer_config.fine_tuning_type+'_'+str(trainer_config.num_epochs)
    
    # Model Saving
    try:
        torch.save(trainer.fine_tuned_model.state_dict(), ckpt_dir / checpkpoint_path)
        logger.info(f"Weights saved to {ckpt_dir.absolute()}")
    except:
        logger.info(f"Unable to save the model")

    # Assessment
    y_pred_train = trainer.predict(X_train)
    y_pred_test = trainer.predict(X_test)
    
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    with open(results_train_dir / f"metrics_{checpkpoint_path}.json", "w") as f:
        json.dump(report_train, f, indent=4)
    
    with open(results_test_dir / f"metrics_{checpkpoint_path}.json", "w") as f:
        json.dump(report_test, f, indent=4)

if __name__ == "__main__":
    main()