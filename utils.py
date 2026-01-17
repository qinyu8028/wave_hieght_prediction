import logging
import os
import matplotlib.pyplot as plt
import sys


def build_logger(log_dir='./log'):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger
    
    log_path = os.path.join(log_dir, 'train.log')
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _plot_loss(train_losses, save_dir='./figures'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_preds(preds, ground_truth, length=300, save_dir='./figures'):
    last_ground_truth = ground_truth[-length:]
    last_preds = preds[-length:]
    title_suffix = f"(Last {length} points)"

    plt.plot(last_ground_truth, label='Ground Truth', color='black', alpha=0.8)
    plt.plot(last_preds, label='Prediction', alpha=0.8, linestyle='--')
    
    plt.title(f'Wave Height Prediction {title_suffix}')
    plt.xlabel('Time Step')
    plt.ylabel('Wave Height (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'prediction_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualization(train_losses, preds, ground_truth, save_dir='./figures', length=100):
    os.makedirs(save_dir, exist_ok=True)
    _plot_loss(train_losses, save_dir=save_dir)
    _plot_preds(preds, ground_truth, length=length, save_dir=save_dir)