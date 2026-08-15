# -*- coding: utf-8 -*-
import copy
import json
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss, mean_squared_error, mean_absolute_error, \
    r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from gplearn.genetic import SymbolicRegressor  # 改为回归器
from gplearn.functions import make_function
from sklearn.multiclass import OneVsRestClassifier
import joblib
from tqdm import tqdm
from factor_eval import _factor_eval_main
import gplearn.fitness
from gplearn.fitness import _Fitness, make_fitness
from gplearn.genetic import BaseSymbolic
from gplearn.fitness import _fitness_map, make_fitness
from sklearn.base import RegressorMixin, ClassifierMixin, TransformerMixin
import cloudpickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import time
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

_timing_details = {}
# 设置随机种子确保结果可复现
torch.manual_seed(1)
np.random.seed(1)


def _ic(y_true, y_pred, w):
    """计算因子值和收益率的秩相关系数（Rank IC）"""
    # 检查是否有日期信息（通过w参数传递）
    if w is None or len(w) != len(y_pred):
        # 如果没有日期信息，使用全局计算
        valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_pred_clean = y_pred[valid_mask]
        y_true_clean = y_true[valid_mask]

        if len(y_pred_clean) < 2:
            return 0.0

        ic_val, _ = spearmanr(y_pred_clean, y_true_clean)
        from scipy.stats import pearsonr
        # ic_val, _ = pearsonr(y_pred_clean, y_true_clean)
        return ic_val if not np.isnan(ic_val) else 0.0

    # 如果有日期信息，使用与compute_ic完全一致的逻辑
    dates = w  # w参数传递日期信息

    # 创建DataFrame并清理数据
    df = pd.DataFrame({
        'factor': y_pred,
        'return': y_true,
        'date': dates
    }).dropna()

    # 如果没有有效数据，返回0
    if df.empty or len(df['date'].unique()) == 0:
        return 0.0

    # 按日期分组计算每日RankIC

    daily_ics = (df.groupby('date')[['factor', 'return']]
                 .apply(lambda g: g['factor'].corr(g['return'], method='pearson')))
    avg_ic = daily_ics.mean()
    return avg_ic if not np.isnan(avg_ic) else 0.0



def _abs_ic(y_true, y_pred, w):
    """计算IC的绝对值（惩罚反向预测）"""
    return np.abs(_ic(y_true, y_pred, w))



class ICFitness(_Fitness):
    def __init__(self):
        super().__init__(_abs_ic, greater_is_better=True)

    def __call__(self, y, y_pred, sample_weight):
        return self.function(y, y_pred, sample_weight)


# 重写SymbolicRegressor核心类
class CustomSymbolicRegressor(SymbolicRegressor):
    def __init__(self,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        # 强制使用IC作为适应度指标
        super().__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=ICFitness(),  # 强制使用IC指标
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )



# 配置参数
config = {
    'train_start': '2016-01-01',
    'train_end': '2020-11-30',
    'inf_start': '2021-01-01',
    'inf_end': '2024-12-27',
    "feature_path": "factor_HFD03_HFD59_bar4_20160101_20241231.h5",
    "label_path": "GP-CNN/label_rq_1Dvwap_0p2class_20100101_20241231.csv.gz",
    "available_path": "GP-CNN/allA_available_20050101_20241231.csv.xz",
    "return_path": "allA_T2_T1_vwap_rq_20100101_20231231.csv.xz",
    'save_dir': './CNN-GP',
    'normalization': 'cs-zscore',
    'patience': 5,
    'max_epoch': 200,
    'batch_size': 518,
    'learning_rate': 0.0005,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cnn_latent_dim': 10,  # CNN提取的特征维度
    'cnn_task': 'classification',  # classification
    'gp_task': 'regression'  # GP任务也改为回归
}

# 定义函数集
funcs_set = ('add', 'sub', 'mul', 'div', 'sin', 'cos', 'tan', 'max', 'min', 'sqrt', 'log', 'abs', 'neg', 'inv')

gp_config = {
    'generations': 1 , # 每个训练轮次只增加1代
    'population_size': 3000,  # 大种群
    'function_set': funcs_set,
    'warm_start': True,  # 启用热启动
    'max_samples': 0.9,  # 使用90%数据训练0.8
    'parsimony_coefficient': 0.01,  # 添加复杂性惩罚0.0001
    'tournament_size': 100,
    'verbose': 0,  # 减少输出
    'random_state': 42,
    'n_jobs': -1,  # 使用所有CPU核心

}


class StockDataset(Dataset):
    def __init__(self, features, labels, dates=None, task='classification'):
        # 确保转换为NumPy数组
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(labels, pd.DataFrame):
            labels = labels.values

        if labels.ndim > 1:
            labels = labels.squeeze()
        print("特征形状:", features.shape)
        print("标签形状:", labels.shape)
        print("标签值分布:", np.unique(labels, return_counts=True))
        self.features = torch.tensor(features, dtype=torch.float32)
        self.dates = dates
        # 根据任务类型处理标签
        self.task = task
        if task == 'regression':
            self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        else:  # classification
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # return self.features[idx], self.labels[idx]
        if self.dates is not None:
            return self.features[idx], self.labels[idx], self.dates[idx]
        else:
            return self.features[idx], self.labels[idx]


def custom_collate_fn(batch):
    """自定义collate函数处理日期"""
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])

    # 检查是否有日期
    if len(batch[0]) == 3:
        dates = [item[2] for item in batch]
        return features, labels, dates
    else:
        return features, labels



class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, feature_length, latent_dim, task='classification'):
        super().__init__()
        self.task = task
        self.input_channels = input_channels  # 添加这行
        self.feature_length = feature_length  # 添加这行

        # 卷积块1
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        # 卷积块2
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.4)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.4)
        )


        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 自适应全局池化
        self.feature_adjust = nn.Conv1d(256, latent_dim, kernel_size=1)

        # 任务特定的输出层
        if task == 'regression':
            # 回归任务输出头
            self.regression_head = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # 回归任务输出1个值
            )
        else:
            # 分类任务输出头（用于训练，但特征提取时不用）
            # self.classification_head = nn.Linear(latent_dim, config.get('num_classes', 3))
            self.classification_head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.Linear(64, config.get('num_classes', 3))  # 分类任务输出多个类别
            )

    def forward(self, x, return_features=False):
        # 输入形状: (batch, features) -> (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # x = x.permute(0, 2, 1)



        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)  # 形状: (batch, 128, 1)
        x = self.feature_adjust(x)  # 形状: (batch, latent_dim, 1)
        features = x.squeeze(-1)  # 形状: (batch, latent_dim)

        # 根据任务返回不同输出
        if return_features:
            return features
        elif self.task == 'regression':
            return self.regression_head(features)
        else:
            return self.classification_head(features)








def safe_zscore(series):
    valid = series.dropna()
    if len(valid) < 2:
        return series
    return (series - series.mean()) / series.std()


def prepare_dataframe_from_raw(conf):
    # 读取原始数据
    df_all = pd.read_hdf(conf['feature_path'], key='data').reset_index()
    # df_all = pd.read_csv(conf['feature_path'], index_col=[0, 1]).reset_index()
    df_all.rename(columns={'day_date': 'date'}, inplace=True)
    df_all['code'] = df_all['code'].str[:6]
    # df_all['code'] = df_all['code'].str[-6:]
    df_all.set_index(['date', 'code'], inplace=True)


    print("处理后的数据列名:", list(df_all.columns))
    print("\n前3行数据示例:")
    print(df_all.head(10))
    print("\n后3行数据示例:")
    print(df_all.tail(10))

    # 分割训练集和推理集
    df_factor = df_all[(df_all.index.get_level_values('date') >= conf['train_start']) &
                       (df_all.index.get_level_values('date') <= conf['train_end'])]
    df_inf = df_all[(df_all.index.get_level_values('date') >= conf['inf_start']) &
                    (df_all.index.get_level_values('date') <= conf['inf_end'])].copy()

    # 读取标签
    df_label = pd.read_csv(conf['label_path'])
    df_label.rename(columns={'datetime': 'date', 'stock_code': 'code'}, inplace=True)
    print("\n标签前3行数据示例:")
    print(df_label.head(10))
    df_label['code'] = df_label['code'].str[:6]
    df_label.set_index(['date', 'code'], inplace=True)
    df_label = df_label[(df_label.index.get_level_values('date') >= conf['train_start']) &
                        (df_label.index.get_level_values('date') <= conf['train_end'])]

    # 读取收益率数据
    returns_df = pd.read_csv(conf['return_path'], index_col=[0])
    returns_df.index.name = 'date'

    # 处理列名：去掉最后3个字符，使其成为6位股票代码
    returns_df.columns = returns_df.columns.map(lambda x: x[:-3])

    # 重塑为多索引格式 (date, code)
    returns_df = returns_df.stack().reset_index()
    returns_df.columns = ['date', 'code', 'return']
    returns_df['date'] = returns_df['date'].astype(str)  # 确保日期为字符串
    returns_df['code'] = returns_df['code'].astype(str).str[:6]  # 确保股票代码为6位字符串
    returns_df.set_index(['date', 'code'], inplace=True)
    returns_df = returns_df[~returns_df.index.duplicated(keep='first')]  # 去除重复索引


    # 取公共索引后先排序
    common_index = df_factor.index.intersection(df_label.index).sort_values()
    df_factor = df_factor.loc[common_index]
    df_label = df_label.loc[common_index]

    # 处理无穷值
    df_factor.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_inf.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Checking for duplicate indices...")
    print(f"df_factor duplicates: {df_factor.index.duplicated().sum()}")
    print(f"df_label duplicates: {df_label.index.duplicated().sum()}")

    # 移除重复索引项（保留第一个出现）
    df_factor = df_factor.loc[~df_factor.index.duplicated(keep='first')]
    df_label = df_label.loc[~df_label.index.duplicated(keep='first')]

    # 确保两个DataFrame的索引完全对齐
    common_index = df_factor.index.intersection(df_label.index)
    print(f"Common index count: {len(common_index)}")

    df_factor = df_factor.loc[common_index]
    df_label = df_label.loc[common_index]
    # 数据标准化
    normalization = conf['normalization']
    if normalization == 'cs-zscore':
        def apply_zscore(group):
            for col in df_factor.columns:
                if group[col].notna().sum() > 1:  # 至少有2个非NaN值
                    group[col] = safe_zscore(group[col])
            return group

        df_factor = df_factor.groupby(level='date', group_keys=False).apply(apply_zscore)
        df_factor = df_factor.fillna(0)

        # 测试集使用当天的均值和标准差
        def apply_zscore_inf(group):
            for col in df_inf.columns:
                if group[col].notna().sum() > 1:  # 至少有2个非NaN值
                    group[col] = safe_zscore(group[col])
            return group
        df_inf = df_inf.groupby(level='date', group_keys=False).apply(apply_zscore_inf)
        df_inf = df_inf.fillna(0)
    elif normalization == 'zscore':
        # 训练集统计量
        train_mean = df_factor.mean()
        train_std = df_factor.std()

        # 训练集标准化
        df_factor = (df_factor - train_mean) / train_std
        df_factor = df_factor.fillna(0)

        # 测试集使用训练集的统计量
        df_inf = (df_inf - train_mean) / train_std
        df_inf = df_inf.fillna(0)

    # 合并因子和标签
    train_data = pd.concat([df_factor, df_label[['label']]], axis=1)
    train_data = train_data.dropna(subset=['label'])

    # 分离特征和标签
    train_feature = train_data.drop(columns=['label']).values
    train_label = train_data[['label']].values




    # 为标签重新编码：从0开始
    unique_labels = sorted(np.unique(train_label))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    config['num_classes'] = len(label_mapping)
    train_label = np.vectorize(label_mapping.get)(train_label)

    # +++ 添加详细的类别分布验证 +++
    print("\n=== 标签分布验证 ===")
    print(f"总样本数: {len(train_label)}")

    # 统计并打印每个原始标签的分布
    raw_label_counts = pd.Series(train_data['label'].values.ravel()).value_counts().sort_index()
    print("\n原始标签分布:")
    print(raw_label_counts.to_string())

    # 统计并打印重新编码后的分布
    encoded_counts = pd.Series(train_label.ravel()).value_counts().sort_index()
    print("\n重新编码后分布:")
    print(encoded_counts.to_string())

    # 验证映射一致性
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    for encoded_label in sorted(reverse_mapping.keys()):
        original_label = reverse_mapping[encoded_label]
        count_original = raw_label_counts[original_label]
        count_encoded = encoded_counts[encoded_label]
        assert count_original == count_encoded, (
            f"标签映射不一致! 原始标签 {original_label} 有 {count_original} 个样本, "
            f"编码标签 {encoded_label} 有 {count_encoded} 个样本"
        )
    print("✓ 标签映射一致性验证通过")










    # 获取收益率数据用于CNN回归任务
    train_return = returns_df.loc[train_data.index]['return'].values.reshape(-1, 1)

    # 保存标签映射关系（用于因子评估）
    with open(f"{conf['save_dir']}/label_mapping.json", 'w') as f:
        json.dump(label_mapping, f)

    # return train_feature, train_label, df_inf, label_mapping, train_return
    train_data = train_data.sort_index(level='date')  # 新增：按日期排序
    train_dates = train_data.index.get_level_values('date').values
    dt = pd.to_datetime(train_dates, format="%Y-%m-%d")
    assert (dt[1:] >= dt[:-1]).all(), "✗ train_dates 非单调递增！"
    print("✓ train_dates 时间顺序 OK")
    return train_feature, train_label, df_inf, label_mapping, train_return, train_dates


def train_cnn_extractor(conf):
    # 准备数据
    prep_start = time.time()
    print("Preparing data...")
    # features, labels, _, label_mapping, returns = prepare_dataframe_from_raw(conf)
    features, labels, _, label_mapping, returns, train_dates = prepare_dataframe_from_raw(conf)
    cnn_data_prep_time = time.time() - prep_start
    # 根据任务类型选择标签
    if conf['cnn_task'] == 'regression':
        # 使用未来收益率作为回归目标
        print("Using returns as regression target")
        regression_targets = returns
    else:
        print("Using classification labels")
        regression_targets = labels

    # 检查数据
    if len(features) == 0:
        raise ValueError("No training data available")


    if conf['cnn_task'] == 'regression':
        # 回归任务使用收益率作为目标
        X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(
            features, regression_targets, train_dates, test_size=0.2, random_state=42
        )





    else:
        # ➤ 1. 把字符串日期转成 datetime64[ns]，再降精度到天
        train_dates_dt = pd.to_datetime(train_dates).values.astype('datetime64[D]')

        # ➤ 2. 用 NumPy 向量化 API 处理
        unique_days = np.unique(train_dates_dt)  # O(N)
        cut = int(len(unique_days) * 0.8)
        train_days, val_days = unique_days[:cut], unique_days[cut:]

        mask = np.isin(train_dates_dt, train_days)  # 彻底跑在 C 层
        X_train, X_val = features[mask], features[~mask]
        y_train, y_val = regression_targets[mask], regression_targets[~mask]
        dates_train, dates_val = train_dates_dt[mask], train_dates_dt[~mask]
        print(f"Total unique days: {len(unique_days)}")
        print(f"Train days: {len(train_days)}")
        print(f"Validation days: {len(val_days)}")



    # 创建数据集
    print("Creating data loaders...")

    train_dataset = StockDataset(X_train, y_train, dates=dates_train, task=conf['cnn_task'])
    val_dataset = StockDataset(X_val, y_val, dates=dates_val, task=conf['cnn_task'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=conf['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn  # 使用自定义collate函数
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=conf['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn  # 使用自定义collate函数
    )

    # 模型初始化
    device = torch.device(conf['device'])
    input_channels = 1
    feature_length = X_train.shape[1]
    #
    model = CNNFeatureExtractor(
        input_channels,
        feature_length,
        conf['cnn_latent_dim'],
        task=conf['cnn_task']
    ).to(device)



    if conf['cnn_task'] == 'regression':
        criterion = nn.MSELoss()  # 回归任务使用均方误差
    else:
        criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失

    optimizer = optim.Adam(
        model.parameters(),
        lr=conf['learning_rate']
    )

    # 训练循环
    best_val_loss = float('inf')
    best_val_metric = -float('inf') if conf['cnn_task'] == 'regression' else 0.0
    best_epoch = -1
    train_losses, val_losses = [], []
    val_metrics = []  # 存储回归指标

    print(f"Starting CNN training ({conf['cnn_task']})...")
    epoch_times = []
    for epoch in range(conf['max_epoch']):
    # for epoch in range(1):
        epoch_start = time.time()

        model.train()
        epoch_train_loss = 0.0

        # 训练批次
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            if len(batch) == 3:  # 有日期信息
                inputs, targets, dates = batch
            else:  # 没有日期信息
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if conf['cnn_task'] == 'regression':
                outputs = model(inputs)
            else:
                # 分类任务需要返回类别概率
                outputs = model(inputs)
                targets = targets.squeeze().long()  # 分类标签需要长整型

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # 验证批次
        model.eval()
        epoch_val_loss = 0.0
        all_preds, all_targets, all_dates = [], [],[]

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                if len(batch) == 3:  # 有日期信息
                    inputs, targets, dates = batch
                    all_dates.extend(dates)
                else:  # 没有日期信息
                    inputs, targets = batch

                inputs, targets = inputs.to(device), targets.to(device)

                if conf['cnn_task'] == 'regression':
                    outputs = model(inputs)

                else:
                    outputs = model(inputs)
                    targets = targets.squeeze().long()  # 分类标签需要长整型

                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

                # 收集预测和实际值用于评估
                if conf['cnn_task'] == 'regression':
                    # all_preds.append(outputs.cpu().numpy())
                    # all_targets.append(targets.cpu().numpy())
                    all_preds.extend(outputs.cpu().numpy().flatten().tolist())
                    all_targets.extend(targets.cpu().numpy().flatten().tolist())
                else:
                    # _, predicted = torch.max(outputs.data, 1)
                    # all_preds.extend(predicted.cpu().numpy())
                    # all_targets.extend(targets.cpu().numpy())
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy().tolist())
                    all_targets.extend(targets.cpu().numpy().tolist())

        # 计算平均损失
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 计算额外评估指标
        metrics = {'val_loss': val_loss}

        if conf['cnn_task'] == 'regression':
            all_preds = np.vstack(all_preds).flatten()
            all_targets = np.vstack(all_targets).flatten()
            if len(all_dates) > 0:
                dates = np.array(all_dates)
            else:
                dates = None

            mse = mean_squared_error(all_targets, all_preds)
            mae = mean_absolute_error(all_targets, all_preds)
            r2 = r2_score(all_targets, all_preds)
            ic_results = compute_ic(
                all_preds.flatten(),
                all_targets.flatten(),
                dates  # 需要确保数据集包含日期
            )

            metrics.update({
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'ic': ic_results['average_rank_IC']  # 存储平均IC值
            })

            # print(f"Epoch {epoch + 1:03d} | "
            #       f"Train Loss: {train_loss:.4f} | "
            #       f"Val Loss: {val_loss:.4f} | "
            #       f"MSE: {mse:.4f} | MAE: {mae:.4f} | "
            #       f"R²: {r2:.4f} | IC: {avg_ic:.4f}")  # 使用浮点数格式化
            print(f"Epoch {epoch + 1:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"MSE: {mse:.4f} | MAE: {mae:.4f} | "
                  f"R²: {r2:.4f} | IC: {ic_results['average_rank_IC']:.4f} | "
                  f"ICIR: {ic_results['ICIR']:.4f}")  # 增加ICIR报告
        else:

            accuracy = accuracy_score(all_targets, all_preds)
            metrics['accuracy'] = accuracy


            cm = confusion_matrix(all_targets, all_preds)



            class_labels = list(label_mapping.keys())
            idx_to_label = {idx: orig_label for orig_label, idx in label_mapping.items()}

            # 获取按索引排序的类别名称
            class_indices = sorted(idx_to_label.keys())
            target_names = [str(idx_to_label[i]) for i in class_indices]

            # 使用classification_report一次性获取所有指标
            report = classification_report(
                all_targets, all_preds,
                # target_names=[str(k) for k in class_labels],
                target_names=target_names,
                output_dict=True,  # 获取结构化数据
                zero_division=0
            )

            # 从报告中提取总体准确率
            accuracy = report['accuracy']
            metrics['accuracy'] = accuracy

            # 精简输出：只打印分类报告
            print(f"\nEpoch {epoch + 1:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {accuracy:.4f}")

            print("\nClassification Report:")
            print(classification_report(
                all_targets, all_preds,
                target_names=[str(k) for k in class_labels],
                zero_division=0
            ))

            # 按类别提取指标用于存储
            per_class_metrics = {}
            for label in class_labels:
                label_str = str(label)
                if label_str in report:
                    per_class_metrics[label] = {
                        'precision': report[label_str]['precision'],
                        'recall': report[label_str]['recall'],
                        'f1-score': report[label_str]['f1-score'],
                        'support': report[label_str]['support']
                    }

            # 存储重要指标
            metrics.update({
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'per_class_metrics': per_class_metrics
            })



        val_metrics.append(metrics)
        current_metric = metrics['ic'] if conf['cnn_task'] == 'regression' else metrics['accuracy']

        if val_loss < best_val_loss and current_metric > best_val_metric:
            best_val_loss = val_loss
            best_val_metric = current_metric
            best_epoch = epoch
            torch.save(model.state_dict(), f"{conf['save_dir']}/best_extractor.pth")
            print(f"Saved best model at epoch {epoch + 1}")

        epoch_times.append(time.time() - epoch_start)

        if epoch - best_epoch >= conf['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('CNN Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 回归任务额外指标可视化
    if conf['cnn_task'] == 'regression':
        plt.subplot(1, 2, 2)
        mses = [m['mse'] for m in val_metrics]
        maes = [m['mae'] for m in val_metrics]
        r2s = [m['r2'] for m in val_metrics]
        ics = [m['ic'] for m in val_metrics]

        plt.plot(mses, 'r-', label='MSE')
        plt.plot(maes, 'g-', label='MAE')
        plt.plot(r2s, 'b-', label='R²')
        plt.plot(ics, 'm-', label='IC')

        plt.title('Regression Metrics')
        plt.xlabel('Epochs')
        plt.legend()
    else:
        # 分类任务绘制混淆矩阵
        model.load_state_dict(torch.load(f"{conf['save_dir']}/best_extractor.pth"))
        model.eval()

        # 获取完整验证集预测
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0], batch[1]
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())



    plt.tight_layout()
    plt.savefig(f"{conf['save_dir']}/extractor_training.png")
    plt.close()

    print(f"Best model at epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}")
    model.load_state_dict(torch.load(f"{conf['save_dir']}/best_extractor.pth"))
    _timing_details['cnn_data_prep'] = cnn_data_prep_time
    _timing_details['cnn_epoch_times'] = epoch_times
    return model


def compute_ic(factor_values, returns, dates):
    """
    与因子评估代码完全一致的IC计算方法
    返回格式: {'ICIR': icir, 'average_rank_IC': avg_ic, 'cumulation_rank_IC': daily_ics}
    """
    # 创建DataFrame并清理数据
    df = pd.DataFrame({
        'factor': factor_values,
        'return': returns,
        'date': dates
    }).dropna()

    # 如果没有有效数据，返回默认值
    if df.empty or len(df['date'].unique()) == 0:
        return {
            'ICIR': 0.0,
            'average_rank_IC': 0.0,
            'cumulation_rank_IC': pd.Series(dtype=float)
        }

    # 按日期分组计算每日RankIC
    # daily_ics = df.groupby('date').apply(
    #     lambda x: x['factor'].corr(x['return'], method='spearman')
    # )
    daily_ics = (df.groupby('date')[['factor', 'return']]
                 .apply(lambda g: g['factor'].corr(g['return'], method='spearman')))
    # 计算各项指标
    avg_ic = daily_ics.mean()
    std_ic = daily_ics.std()
    icir = avg_ic / std_ic if std_ic > 0 else 0.0

    return {
        'ICIR': icir,
        'average_rank_IC': avg_ic,
        'cumulation_rank_IC': daily_ics
    }

def extract_features(model, data, conf):
    """使用训练好的CNN提取特征

    - 小数据（推理每日 ~5000 条）：直接一次性张量前向传播，零开销
    - 大数据（训练全量 ~百万条）：分批处理避免OOM，但 num_workers=0 避免进程创建开销

    原实现用 DataLoader(num_workers=4)，在 Windows 上每次 spawn 4 个 Python 进程。
    推理时 ~1000 天 × 4 workers = 4000 次进程创建，这是最大的时间消耗来源。
    """
    device = torch.device(conf['device'])
    model.eval()

    # 检查数据类型
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise ValueError("Unsupported data format for feature extraction")

    # 推理日数据量小（~5000行），直接一次性前向传播，不经过 DataLoader
    if len(data) <= 50000:
        X_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        with torch.no_grad():
            features = model(X_tensor, return_features=True).cpu().numpy()
        return features

    # 训练阶段全量数据大，分批处理但不用 worker 进程
    dataset = StockDataset(data, np.zeros(len(data)))
    loader = DataLoader(
        dataset,
        batch_size=conf['batch_size'],
        shuffle=False,
        num_workers=0   # ← 关键：设为 0，不启动子进程
    )
    features_list = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            features = model(inputs, return_features=True).cpu().numpy()
            features_list.append(features)
    return np.vstack(features_list)



def train_gp_regressor(conf, gp_conf, cnn_features, returns,train_dates):
    """训练GP回归器 - 使用IC作为早停标准"""
    if returns.ndim > 1:
        returns = returns.squeeze()




    # ---------- 1) 日期向量化：先转成 datetime64[ns] → 再降到按天精度 ----------
    train_dates_dt = pd.to_datetime(train_dates).values.astype('datetime64[D]')

    # ---------- 2) 去重 + 排序 ----------
    unique_days = np.unique(train_dates_dt)  # O(N) 矢量化
    cut = int(len(unique_days) * 0.8)  # 80% 早期做训练
    train_days, val_days = unique_days[:cut], unique_days[cut:]

    # ---------- 3) 掩码切分 ----------
    mask = np.isin(train_dates_dt, train_days)  # 彻底跑在 C 层
    X_train, X_val = cnn_features[mask], cnn_features[~mask]
    r_train, r_val = returns[mask], returns[~mask]
    dates_train, dates_val = train_dates_dt[mask], train_dates_dt[~mask]

    # ---------- 4) 转换日期为数值（时间戳） ----------
    date_codes_train = dates_train.view('int64')  # 转为时间戳 (int64)
    date_codes_val = dates_val.view('int64')  # 同样处理验证集
    print(np.unique(dates_train))
    # 创建新模型
    # gp_model = CustomSymbolicRegressor(**gp_conf)
    gp_model = CustomSymbolicRegressor(
        population_size=gp_conf['population_size'],
        generations=gp_conf['generations'],
        tournament_size=gp_conf['tournament_size'],
        function_set=gp_conf['function_set'],
        parsimony_coefficient=gp_conf['parsimony_coefficient'],
        warm_start=gp_conf['warm_start'],
        max_samples = gp_conf['max_samples'],
        n_jobs=gp_conf['n_jobs'],
        verbose = gp_conf['verbose'],
        random_state=gp_conf['random_state']
    )
    gp_model.train_dates = train_dates
    gp_model.log_fitness = True
    open("fitness_log.csv", "w").close()
    patience = conf['patience']
    best_ic = 0.0  # IC范围[-1, 1]
    best_mse = float('inf')
    # best_model = None
    best_model = gp_model  # 设置初始模型
    no_improve_epochs = 0
    best_gen = 0  # 记录找到最佳个体的代
    gp_model.train_dates = train_dates
    print("Starting GP regression training with IC early stopping...")
    at_least_one_valid = False
    # gp_model.train_dates = dates_train
    fitness_history = []
    gen_times = []
    # 早停训练
    for gen in range(conf['max_epoch']):
        gen_start = time.time()

        gp_model.fit(X_train, r_train, sample_weight=date_codes_train)  # 使用 int64 时间戳
        # gp_model.fit(X_train, r_train, sample_weight=dates_int_train)

        # 在验证集上预测
        y_val_pred = gp_model.predict(X_val)

        # 计算回归指标
        current_mse = mean_squared_error(r_val, y_val_pred)


        ic_results = compute_ic(
            y_val_pred,
            r_val,
            dates_val  # 需要确保验证集有日期
        )

        # 使用相同指标
        current_ic = abs(ic_results['average_rank_IC'])
        icir = ic_results['ICIR']
        print(f"Generation {gen + 1} - "
              f"Validation MSE: {current_mse:.6f}, "
              f"Avg Rank IC(abs): {current_ic:.4f}, "
              f"ICIR: {icir:.4f}")


        improvement = False
        if not at_least_one_valid:
            best_mse = current_mse
            best_model = copy.deepcopy(gp_model)
            best_gen = gen + 1
            improvement = True
            at_least_one_valid = True
            print(f"Initial valid model at generation {gen + 1}")
        elif current_mse < best_mse:  # 仅当MSE更小时视为改进
            best_mse = current_mse
            best_model = copy.deepcopy(gp_model)
            no_improve_epochs = 0
            best_gen = gen + 1
            improvement = True
        if improvement:
            print(f"↑ Improved! Best IC: {best_ic:.4f}, Best MSE: {best_mse:.6f}")
        else:
            no_improve_epochs += 1
            print(f"Generation {gen + 1}/{conf['max_epoch']} - No improvement ({no_improve_epochs}/{patience})")

            # 检查早停条件 - 添加额外的条件确保至少训练一代
        gen_times.append(time.time() - gen_start)
        if no_improve_epochs >= patience and gen >= 1:  # 至少训练一代
            print(f"\nEarly stopping at generation {gen + 1} (Best IC: {best_ic:.4f} at generation {best_gen})")
            break
        gen_fitness = [ind.fitness_ for ind in gp_model._programs[-1]]

        print(f"Gen {gen + 1:02d} Stats → "
              f"Best {max(gen_fitness):.4f}  "
              f"Mean {np.mean(gen_fitness):.4f}  "
              f"Worst {min(gen_fitness):.4f}")



    pd.DataFrame(fitness_history).to_csv(f"{conf['save_dir']}/gp_fitness_history.csv", index=False)
    if not at_least_one_valid:
        # 如果还没有有效模型，使用最后一个模型
        print("\nWarning: No valid model found during training. Using the last model.")
        best_model = gp_model
    # 最终评估
    print(f"\n{'=' * 50}")
    print(f"Best GP Individual Found at Generation {best_gen}:")
    print(f"Validation MSE: {best_mse:.6f}")
    print(f"Validation IC: {best_ic:.4f}")
    print(f"{'=' * 50}")

    # 计算回归指标
    y_val_pred = best_model.predict(X_val)
    mae = mean_absolute_error(r_val, y_val_pred)
    r2 = r2_score(r_val, y_val_pred)

    print("\nRegression Metrics:")
    print(f"MSE: {best_mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.4f}")

    # 保存模型
    # joblib.dump(best_model, f"{conf['save_dir']}/gp_regressor.joblib")
    with open(f"{conf['save_dir']}/gp_regressor.joblib", 'wb') as f:
        cloudpickle.dump(best_model, f)
    print(f"Model saved to {conf['save_dir']}/gp_regressor.joblib")

    # 输出公式
    print("\nBest Formula:")
    print(best_model._program)

    _timing_details['gp_gen_times'] = gen_times
    return best_model


def hybrid_infer_regression(conf, cnn_model, gp_model):
    _, _, df_inf, _, _ ,_= prepare_dataframe_from_raw(conf)
    dates = df_inf.index.get_level_values('date').unique()
    results = []  # 存储每日的因子DataFrame
    # scaler = joblib.load(f"{conf['save_dir']}/cnn_feature_scaler.joblib")
    # 处理每个交易日
    for date in tqdm(dates, desc="Processing Inference Dates"):
        try:
            daily_data = df_inf.loc[date].copy()
            if len(daily_data) == 0:
                continue

            codes = daily_data.index.tolist()
            daily_features = extract_features(cnn_model, daily_data.values, conf)

            # daily_features = scaler.transform(daily_features)

            # 使用GP回归模型预测连续值作为因子
            factor_values = gp_model.predict(daily_features)

            # date_str = pd.to_datetime(date).strftime('%Y%m%d')
            date_str = str(date)
            df_day = pd.DataFrame(factor_values.reshape(1, -1),
                                  index=[date_str],
                                  columns=codes)
            results.append(df_day)

        except Exception as e:
            print(f"Error processing {date}: {str(e)}")
            continue

    if results:
        factor_df = pd.concat(results)
        output_path = f"{conf['save_dir']}/infer_{conf['inf_start']}_{conf['inf_end']}.csv"
        factor_df.to_csv(output_path)
        print(f"Saved factor results to {output_path}")
        return factor_df
    else:
        print("No inference results generated")
        return pd.DataFrame()


def _write_timing_report(conf, timing):
    """将计时数据写入独立文件"""
    report_path = f"{conf['save_dir']}/timing_report.txt"
    total = timing.get('total', 0.0)

    def fmt(seconds):
        """格式化时间为 h m s 格式"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = seconds % 60
            return f"{m}m {s:.1f}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}h {m}m {s:.1f}s"

    def pct(phase_time):
        """计算百分比"""
        return (phase_time / total * 100) if total > 0 else 0.0

    lines = []
    lines.append("=" * 60)
    lines.append("           运行时间统计报告")
    lines.append("=" * 60)
    lines.append("")

    # 总时间
    lines.append(f"  总运行时间:              {fmt(total)}")
    lines.append("")

    # 数据准备
    data_prep = _timing_details.get('cnn_data_prep', 0.0)
    lines.append(f"  数据准备:                 {fmt(data_prep)}  ({pct(data_prep):.1f}%)")

    # CNN 训练
    cnn_time = timing.get('cnn_training', 0.0)
    cnn_epochs = _timing_details.get('cnn_epoch_times', [])
    lines.append(f"  CNN训练:                  {fmt(cnn_time)}  ({pct(cnn_time):.1f}%)")
    if cnn_epochs:
        avg_ep = sum(cnn_epochs) / len(cnn_epochs)
        lines.append(f"    ├─ Epochs: {len(cnn_epochs)}")
        lines.append(f"    ├─ 平均每Epoch: {fmt(avg_ep)}")
        lines.append(f"    ├─ 最快Epoch: {fmt(min(cnn_epochs))}")
        lines.append(f"    └─ 最慢Epoch: {fmt(max(cnn_epochs))}")

    # CNN 特征提取
    feat_time = timing.get('feature_extraction', 0.0)
    lines.append(f"  CNN特征提取:              {fmt(feat_time)}  ({pct(feat_time):.1f}%)")

    # GP 训练
    gp_time = timing.get('gp_training', 0.0)
    gp_gens = _timing_details.get('gp_gen_times', [])
    lines.append(f"  GP训练:                   {fmt(gp_time)}  ({pct(gp_time):.1f}%)")
    if gp_gens:
        avg_gen = sum(gp_gens) / len(gp_gens)
        lines.append(f"    ├─ Generations: {len(gp_gens)}")
        lines.append(f"    ├─ 平均每代: {fmt(avg_gen)}")
        lines.append(f"    ├─ 最快一代: {fmt(min(gp_gens))}")
        lines.append(f"    └─ 最慢一代: {fmt(max(gp_gens))}")

    # 混合推理
    inf_time = timing.get('hybrid_inference', 0.0)
    lines.append(f"  混合推理:                 {fmt(inf_time)}  ({pct(inf_time):.1f}%)")

    # 因子评估
    eval_time = timing.get('factor_evaluation', 0.0)
    lines.append(f"  因子评估:                 {fmt(eval_time)}  ({pct(eval_time):.1f}%)")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n计时报告已保存至: {report_path}")
    print(report)


def main():
    conf = config
    total_start = time.time()
    timing = {}  # 存储所有计时数据

    # 创建保存目录
    os.makedirs(conf['save_dir'], exist_ok=True)
    # 保存配置
    with open(f"{conf['save_dir']}/hybrid_config.json", "w") as f:
        json.dump({"config": conf, "gp_config": gp_config}, f, indent=4)
        print("Saved configuration")

    # === 1. 训练CNN特征提取器（回归任务）===
    phase_start = time.time()
    print("\n" + "=" * 50)
    print("Training CNN Feature Extractor as REGRESSION model".center(50))
    print("=" * 50)
    cnn_model = train_cnn_extractor(conf)

    # 保存CNN模型
    torch.save(cnn_model.state_dict(), f"{conf['save_dir']}/cnn_feature_extractor_regression.pth")
    print("Saved CNN feature extractor (regression)")
    timing['cnn_training'] = time.time() - phase_start

    # === 2. 提取特征用于GP训练 ===
    phase_start = time.time()
    print("\n" + "=" * 50)
    print("Extracting Features for GP Training".center(50))
    print("=" * 50)
    # features, _, _, _, returns = prepare_dataframe_from_raw(conf)
    features, labels, _, _, returns, train_dates = prepare_dataframe_from_raw(conf)
    # 提取特征 - 只使用CNN的特征提取部分
    cnn_features = extract_features(cnn_model, features, conf)
    print(f"Extracted features shape: {cnn_features.shape}")
    timing['feature_extraction'] = time.time() - phase_start


    # === 3. 训练GP回归器 ===
    phase_start = time.time()
    print("\n" + "=" * 50)
    print("Training GP Regressor with IC Early Stopping".center(50))
    print("=" * 50)
    gp_model = train_gp_regressor(conf, gp_config, cnn_features, returns, train_dates)
    timing['gp_training'] = time.time() - phase_start

    # === 4. 执行混合推理 ===
    phase_start = time.time()
    print("\n" + "=" * 50)
    print("Running Hybrid Inference".center(50))
    print("=" * 50)
    factor_df = hybrid_infer_regression(conf, cnn_model, gp_model)
    timing['hybrid_inference'] = time.time() - phase_start

    # === 5. 因子评估 ===
    phase_start = time.time()
    print("\n" + "=" * 50)
    print("Running Factor Evaluation".center(50))
    print("=" * 50)
    _factor_eval_main(config)
    timing['factor_evaluation'] = time.time() - phase_start

    # 计算总时间并写入报告文件
    timing['total'] = time.time() - total_start
    _write_timing_report(conf, timing)

    print("\nAll processes completed successfully!")


if __name__ == "__main__":
    main()
