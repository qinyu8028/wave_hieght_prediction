from model import Model
import torch
from data_process import build_dataloader
import traceback
from utils import build_logger, visualization
import time
import os


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    logger = build_logger(exp_dir)
    logger.info(f"Experiment started. Saving results to: {exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    try:
        logger.info("Loading data...")
        train_loader, test_loader, scaler_Y = build_dataloader(data_path="ocean_buoy_data_june_2023.csv", 
                                                               window_size=12, 
                                                               batch_size=64)
        dataloader = {"train_loader": train_loader, "test_loader": test_loader}

        model = Model(device=device, logger=logger)

        logger.info("----------Starting training----------")
        model.train_model(dataloader=dataloader)

        logger.info("----------Start testing----------")
        preds_original, ground_truth_original = model.test_model(dataloader=dataloader, scaler=scaler_Y)

        logger.info("\n----------Visualizing results----------")
        visualization(train_losses=model.train_losses, preds=preds_original, ground_truth=ground_truth_original)

        logger.info("All done successfully!")

    except Exception as e:
        print(f"Error type: {type(e)}")
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()