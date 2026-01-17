from logging import Logger
from backbone import GRU
from tqdm import tqdm
import torch
import os
from torch import optim, nn, Tensor
from typing import Callable, Dict, List, Tuple
from torch.optim.optimizer import Optimizer
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error


class Model:
    def __init__(
        self, 
        input_size=7, 
        hidden_size=64, 
        output_size=1, 
        logger: Logger =None, 
        criterion=nn.MSELoss(), 
        gru_dropout=0.0, 
        fc_dropout=0.0, 
        num_layers=1, 
        lr=0.001, 
        scheduler_step_size=25, 
        scheduler_gamma=0.5, 
        weight_decay=0.01, 
        device='cuda'
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.logger = logger

        self.model = GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            num_layers=num_layers, 
            gru_dropout=gru_dropout, 
            fc_dropout=fc_dropout
        ).to(device)

        self.model.init_params()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = criterion

        # 训练记录初始化
        self.best_loss = float('inf')
        self.log_train = {}
        self.log_test = {}
        self.train_losses = []

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )


    def _train_one_epoch(self, 
                         log: Dict, 
                         model: nn.Module, 
                         dataloader: DataLoader, 
                         optimizer: Optimizer, 
                         criterion: Callable):
        model = model.train()
        losses = []
        for features, targets in tqdm(dataloader):
            # 传入数据
            features = features.to(self.device)
            targets = targets.to(self.device)
            # 优化器初始化
            optimizer.zero_grad()
            # 前向传播
            outputs = model(features)

            # if outputs.shape != targets.shape:
            #     outputs = outputs.view_as(targets)

            # 计算loss
            loss: Tensor = criterion(outputs, targets)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 记录batch loss
            losses.append(loss.item())

        loss = np.mean(losses)
        log['loss'] = loss
        
        return


    def _predict(self, model: nn.Module, dataloader: DataLoader) -> Tuple[List, List]:
        model = model.eval()
        preds = []
        ground_truth = []
        with torch.no_grad():
            for features, targets in tqdm(dataloader):
                features = features.to(self.device)
        
                prediction: Tensor = model(features)

                preds.append(prediction.cpu().numpy())
                ground_truth.append(targets)
                
        all_preds = np.concatenate(preds)
        all_ground_truth = np.concatenate(ground_truth)

        return all_preds, all_ground_truth


    def _evaluate(self, preds_original, ground_truth_original, scaler: MinMaxScaler):
        # 计算 metrics
        mse = mean_squared_error(ground_truth_original, preds_original)
        rmse = root_mean_squared_error(ground_truth_original, preds_original)
        mae = mean_absolute_error(ground_truth_original, preds_original)
        r2 = r2_score(ground_truth_original, preds_original)

        return mse, rmse, mae, r2


    def train_model(self, dataloader: DataLoader, n_epochs=100, save_dir='./checkpoints', save_interval=10):

        for epoch in range(n_epochs):
            self._train_one_epoch(
                log=self.log_train,
                model=self.model, 
                dataloader=dataloader['train_loader'], 
                optimizer=self.optimizer, 
                criterion=self.criterion
            )

            loss = self.log_train['loss']
            self.train_losses.append(loss)

            self.lr_scheduler.step()

            # 定时保存
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch=epoch, save_dir=save_dir)
                print(f"Checkpoint saved at epoch {epoch+1}, loss: {loss:.6f}")
        
        self.logger.info("----------Training finished----------\n")
        return

    
    def test_model(self, dataloader: DataLoader, scaler: MinMaxScaler, load_path='./checkpoints/best_checkpoint.pth'):
        _, _ = self._select_best_checkpoint(dataloader=dataloader, scaler=scaler)

        # 加载 best_model
        model = torch.load(
            load_path, 
            weights_only=True, 
            map_location=self.device
        )
        self.model.load_state_dict(model)

        self.logger.info("\n----------Starting testing----------")

        preds, ground_truth = self._predict(
            model=self.model,
            dataloader=dataloader['test_loader']
        )

        # 反归一化
        preds_original = scaler.inverse_transform(preds)
        ground_truth_original = scaler.inverse_transform(ground_truth)

        mse, rmse, mae, r2 = self._evaluate(preds_original, ground_truth_original, scaler)

        self.logger.info(f"\nTest Results:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}")

        return preds_original, ground_truth_original


    def save_checkpoint(self, epoch, save_dir='./checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'epoch_{epoch+1}.pth')

        torch.save(self.model.state_dict(), save_path)
        
        return


    def _select_best_checkpoint(self, dataloader: DataLoader, scaler: MinMaxScaler, load_dir='./checkpoints'):
        import glob
        import shutil

        self.logger.info("----------Selecting best checkpoint----------")

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {load_dir}")

        checkpoints = sorted(glob.glob(os.path.join(load_dir, 'epoch_*.pth')))

        if not checkpoints:
            raise ValueError(f"No checkpoint files found in {load_dir}")

        results= {}
        for ckpt_pth in checkpoints:
            model = torch.load(
                ckpt_pth, 
                weights_only=True, 
                map_location=self.device
            )
            self.model.load_state_dict(model)

            preds, ground_truth = self._predict(
                model=self.model,
                dataloader=dataloader['test_loader']
            )

            # 反归一化
            preds_original = scaler.inverse_transform(preds)
            ground_truth_original = scaler.inverse_transform(ground_truth)

            _, rmse, _, _ = self._evaluate(preds_original, ground_truth_original, scaler)

            results[ckpt_pth] = rmse
            self.logger.info(f"{os.path.basename(ckpt_pth)}: RMSE={rmse:.4f}")
        
        best_ckpt = min(results, key=results.get)
        best_rmse = results[best_ckpt]

        best_path = os.path.join(load_dir,'best_checkpoint.pth')
        shutil.copy(best_ckpt, best_path)
        self.logger.info(f"Best model copied to {best_path}. RMSE:{best_rmse:.4f}")

        return best_ckpt, results