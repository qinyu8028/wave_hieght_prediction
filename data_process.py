import pandas as pd
import torch
# from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class WaveDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_cols: list, target_col: str, window_size=12):
        super().__init__()
        self.window_size = window_size
        
        features_list = []
        target_list = []

        for buoy_id, group in df.groupby("buoy_id"):
            data = group[input_cols].values # 这里的data是numpy数组
            target_idx = input_cols.index(target_col)

            for i in range(len(data) - window_size):
                x_window = data[i : i + window_size, :]
                y_label = data[i + window_size, target_idx]

                features_list.append(x_window)
                target_list.append(y_label)
    
        self.features = torch.from_numpy(np.array(features_list)).float()
        self.target = torch.from_numpy(np.array(target_list).reshape(-1, 1)).float()    # 这里y的格式需要注意！需要进行reshape才能转化为tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]


def build_dataloader(data_path="ocean_buoy_data_june_2023.csv", 
                     window_size=12, 
                     batch_size=64) -> tuple[DataLoader, DataLoader, MinMaxScaler]:
    '''
    1. 读取csv数据
    2. 数据清洗：时间转换和排序
    3. 划分数据集和测试集
    4. 数据归一化
    5. 分组滑窗切片, 构造输入输出对
    6. 转Tensor
    7. 封装loader
    '''
    
    df = pd.read_csv(data_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(by=['buoy_id', 'timestamp'])

    df = time_embedding(df)

    # 划分训练集与测试集
    split_date = pd.Timestamp('2023-06-26 00:00:00')
    train_df = df[df['timestamp'] < split_date].copy()
    test_df  = df[df['timestamp'] >= split_date].copy()

    # 归一化
    feature_cols = [
        'latitude', 
        'longitude',
        'wind_speed', 
        'sea_level_pressure', 
        'significant_wave_height'
    ]
    target_col = ['significant_wave_height'] 

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(train_df[feature_cols])
    scaler_Y = MinMaxScaler(feature_range=(0, 1))   # 便于直接恢复数据
    scaler_Y.fit(train_df[target_col])

    train_df[feature_cols] = scaler_X.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler_X.transform(test_df[feature_cols])

    input_cols = [
        'latitude', 
        'longitude',
        'wind_speed', 
        'sea_level_pressure', 
        'significant_wave_height',
        'hour_sin', 
        'hour_cos'
    ]
    target_col_name = 'significant_wave_height'

    train_dataset = WaveDataset(train_df, input_cols, target_col_name, window_size)
    test_dataset = WaveDataset(test_df, input_cols, target_col_name, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, scaler_Y


def time_embedding(df: pd.DataFrame) -> pd.DataFrame:
    hour = df['timestamp'].dt.hour
    
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    return df


if __name__ == '__main__':
    data_path = "ocean_buoy_data_june_2023.csv"
    train_loader, test_loader, scaler_Y = build_dataloader(data_path, window_size=12, batch_size=64)