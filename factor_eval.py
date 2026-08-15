import json
import logging
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 计算IC和RankIC
def calculate_ic_and_rank_ic(factor_df: pd.DataFrame, returns_df: pd.DataFrame):
    ic_values = []  # 存储每个日期的 IC
    rank_ic_values = []  # 存储每个日期的 rank(IC)
    dates = []

    # 遍历因子表中的日期
    for date in factor_df.index.unique():
        # 获取对应日期的因子数据和收益率数据
        factor_data = factor_df.loc[date].dropna()  # 对缺失值进行清理
        return_data = returns_df.loc[date].dropna()  # 对缺失值进行清理

        # 保证因子数据和收益率数据的股票数量一致
        common_stocks = factor_data.index.intersection(return_data.index)

        # 如果有共同的股票，则计算 IC 和 rank(IC)
        if len(common_stocks) > 0:
            factor_data_common = factor_data[common_stocks].values.flatten()
            return_data_common = return_data[common_stocks].values.flatten()

            # 计算 Pearson 相关系数（IC）
            ic = pd.Series(factor_data_common).corr(pd.Series(return_data_common))
            ic_values.append(ic)

            # 计算 rank(IC)，首先对因子和收益率进行排名
            #factor_rank = pd.Series(factor_data_common).rank()
            #return_rank = pd.Series(return_data_common).rank()

            # 计算排名之间的相关性
            #rank_ic = factor_rank.corr(return_rank)
            rank_ic = pd.Series(factor_data_common).corr(pd.Series(return_data_common), method='spearman')
            rank_ic_values.append(rank_ic)

        else:
            ic_values.append(None)
            rank_ic_values.append(None)

        dates.append(date)

    # 计算平均 IC 和 rank(IC)
    avg_ic = pd.Series(ic_values).mean()
    std_ic = pd.Series(ic_values).std()

    avg_rank_ic = pd.Series(rank_ic_values).mean()

    # 输出结果
    return {'ICIR': avg_ic / std_ic, 'average_rank_IC': avg_rank_ic,
            'cumulation_rank_IC': pd.Series(rank_ic_values, index=dates)}

def calculate_rank_ic_v2(factor_df: pd.DataFrame, returns_df: pd.DataFrame):
    # 计算IC和RankIC
    icresult = factor_df.corrwith(returns_df, method='spearman', axis=1)
    rankic = icresult.mean()
    icir = icresult.mean() / icresult.std()
    icresults = {'ICIR': icir, 'average_rank_IC': rankic,
            'cumulation_rank_IC': pd.Series(icresult)}

    return icresults

def AnnualizedReturnFun(Ret, N):
    return (1 + Ret) ** (252 / N) - 1



def plot_factor_group_return_v2(df_F_smooth_n: pd.DataFrame, df_R: pd.DataFrame, group_num: int):
    """
    Plot the annualized return for each group if we divide all stocks into group_num groups.
    This version is faster than the original version.
    """

    def AnnualizedReturnFun(Ret, N):
        AR = (1 + Ret) ** (252 / N) - 1
        return AR

    df_R_check = df_R.reindex(columns=df_F_smooth_n.columns)
    df_R_check = df_R_check.loc[df_F_smooth_n.index]
    df_F_vals = df_F_smooth_n.values
    df_R_vals = df_R_check.fillna(0).values
    # df_R_vals = df_R.values
    N = df_F_smooth_n.shape[0]
    res = np.empty((N, group_num))
    res[:] = np.nan
    for i in range(N):
        cur_F = df_F_vals[i, :]
        cur_R = df_R_vals[i, :]
        if i < 3:  # 只打印前3天的日志
            print(f"\n日期 {df_F_smooth_n.index[i]}")
            print(f"因子值: {cur_F[:5]}")
            print(f"收益率值: {cur_R[:5]}")

        cur_F_R = np.vstack([cur_F, cur_R])
        cur_F_R = cur_F_R[:, ~np.isnan(cur_F_R[0, :])]
        cur_F_R = cur_F_R[:, (-cur_F_R[0, :]).argsort()]
        cur_N = cur_F_R.shape[1]
        stk_num = cur_N // group_num
        temp = []
        for which_group in range(1, group_num + 1):
            B = cur_F_R[1, ((which_group - 1) * stk_num):(which_group * stk_num)]
            C = np.nanmean(B)
            temp.append(C)
        res[i, :] = temp
    df = pd.DataFrame(res)
    cum_rtn = (1 + df.fillna(0)).cumprod().iloc[-1] - 1
    ann_rtn = AnnualizedReturnFun(cum_rtn, N)
    ann_rtn.name = 'grouped_annualized_return'
    return ann_rtn

# 计算交易成本后的因子收益
def cal_single_factor_topBottom_rtn_wcost(df: pd.DataFrame, df_R: pd.DataFrame, up=90, low=10):
    """
    Calculate the single factor_rank return with the transaction cost included.
    """

    def retain_extreme_10_percent(row):
        row = row.copy()
        row -= np.min(row)  # 确保打分>0
        sorted_row = np.sort(row.dropna())  # 排序并忽略NaN值
        if len(sorted_row) == 0:
            return row  # 如果行全是NaN，则不做处理直接返回

        low_threshold = np.percentile(sorted_row, low)  # 下10%的阈值
        high_threshold = np.percentile(sorted_row, up)  # 上10%的阈值

        new_row = pd.Series(np.nan, index=row.index)

        # 计算权重并赋值
        new_row[row <= low_threshold] = 1 / len(row[row <= low_threshold]) * -0.5
        new_row[row >= high_threshold] = 1 / len(row[row >= high_threshold]) * 0.5

        # 将不在前后10%数值区间内的值设为NaN
        return new_row

    df_F_n = df.apply(retain_extreme_10_percent, axis=1)
    df_F_n = df_F_n.fillna(0)
    df_R_check = df_R.reindex(columns=df.columns)
    df_R_check = df_R_check.loc[df_F_n.index]
    res = (df_F_n * df_R_check).sum(axis=1)
    nav = 1 + res
    nav = nav.cumprod()
    return res, nav, df_F_n


# 计算换手率
def evalu_turnover(position_df):
    daily_change = position_df.diff().abs()
    daily_turnover = daily_change.sum(axis=1)
    average_turnover = daily_turnover.mean()
    return average_turnover


# 计算多头超额收益
def cal_factor_top_extra_rtn(df: pd.DataFrame, df_R: pd.DataFrame, up=90, pool_df=None):
    '''用中证全指计算多头超额，取因子top10%'''

    def retain_extreme_10_percent(row):
        row = row.copy()
        row -= np.min(row)  # 确保打分>0
        sorted_row = np.sort(row.dropna())  # 排序并忽略NaN值
        if len(sorted_row) == 0:
            return row  # 如果行全是NaN，则不做处理直接返回

        high_threshold = np.percentile(sorted_row, up)  # 上10%的阈值
        # low_threshold = np.percentile(sorted_row, 85)  # 上10%的阈值
        new_row = pd.Series(np.nan, index=row.index)
        # 计算权重并赋值
        new_row[(row >= high_threshold)] = 1 / len(row[(row >= high_threshold)])  # 等权持有
        # 将不在前后10%数值区间内的值设为NaN
        return new_row

    df_F_n = df.apply(retain_extreme_10_percent, axis=1)
    df_F_n = df_F_n.fillna(0)
    df_R_check = df_R.reindex(columns=df.columns)
    df_R_check = df_R_check.loc[df_F_n.index]
    res = (df_F_n * df_R_check).sum(axis=1)
    if pool_df is not None:
        base_rtn = (df_R * pool_df).mean(axis=1)
    else:
        base_rtn = df_R.mean(axis=1)
    extra_rtn = res - base_rtn
    nav_df = pd.DataFrame()
    nav_df['extra_nav'] = (1 + extra_rtn).cumprod()
    final_extra_rtn = nav_df['extra_nav'].values[-1] - 1
    annual_extra_rtn = (1 + final_extra_rtn) ** (252 / len(nav_df)) - 1
    sharpe_based_on_index = annual_extra_rtn / (extra_rtn.std() * np.sqrt(252))

    long_turnover = evalu_turnover(df_F_n)
    return {'nav': nav_df, 'ann_rtn': annual_extra_rtn, 'index_sharpe': sharpe_based_on_index, 'turnover': long_turnover}

def load_dfs(conf):
    start_date, end_date = conf['inf_start'], conf['inf_end']
    factor_df = pd.read_csv(f"{conf['save_dir']}/infer_{conf['inf_start']}_{conf['inf_end']}.csv", index_col=[0])
    # returns_df = pd.read_csv(conf['return_path'], index_col=[0])
    # factor_df = factor_df[(factor_df.index >= start_date) & (factor_df.index <= end_date)]
    # returns_df = returns_df[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    bar_available = pd.read_csv(conf['available_path'], index_col=0)
    bar_available.columns = bar_available.columns.map(lambda x: x[:-3])
    bar_available.index.name='date'
    bar_available = bar_available.reindex_like(factor_df)
    returns_df = pd.read_csv(conf['return_path'], index_col=[0])
    factor_df = factor_df[(factor_df.index >= start_date) & (factor_df.index <= end_date)]
    returns_df = returns_df[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    factor_df = factor_df * bar_available

    returns_df.index.name='date'
    returns_df.columns = returns_df.columns.map(lambda x: x[:-3])

    # 确保因子和收益率的股票列一致
    common_stocks = factor_df.columns.intersection(returns_df.columns)
    factor_df = factor_df[common_stocks]
    returns_df = returns_df[common_stocks]

    # 确保因子数据和收益率数据日期对齐，保留共同的日期
    common_dates = factor_df.index.intersection(returns_df.index)
    factor_df = factor_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]






    return factor_df, returns_df





def plot_all(factor_df: pd.DataFrame, returns_df: pd.DataFrame, save_dir=None, description='',):
    # 计算IC和RankIC






    icresults = calculate_rank_ic_v2(factor_df, returns_df)
    logging.info(f"Average IR: {icresults['ICIR']}")
    logging.info(f"Average Rank(IC): {icresults['average_rank_IC']}")

    factor_df = np.sign(icresults['average_rank_IC']) * factor_df
    #returns_df = returns_df.fillna(0)

    # 计算年化收益率，假设分为10组
    group_num = 10
    annualized_returns = plot_factor_group_return_v2(factor_df, returns_df, group_num)

    # 计算多头超额收益
    extra_rtn_results = cal_factor_top_extra_rtn(factor_df, returns_df)

    # 计算交易成本后的因子收益
    factor_rtn, nav, adjusted_factor = cal_single_factor_topBottom_rtn_wcost(factor_df, returns_df)

    # 计算换手率
    turnover = evalu_turnover(adjusted_factor)

    # 输出结果
    logging.info(f"多空收益率: {AnnualizedReturnFun(nav[-1]-1,len(nav))}")
    logging.info(f"每组年化收益率:\n {annualized_returns}")
    logging.info(f"换手率: {turnover}")
    logging.info(f"多头超额收益（年化）：{extra_rtn_results['ann_rtn']}")
    logging.info(f"多头超额收益（年化）/波动率：{extra_rtn_results['index_sharpe']}")
    logging.info(f"多头超额收益（累计净值）：{extra_rtn_results['nav'].iloc[-1]['extra_nav']}")
    logging.info(f"多头换手率: {extra_rtn_results['turnover']}")



    # 设置字体和字体大小
    # 获取所有字体
    font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    font_names = [font_manager.FontProperties(fname=fname).get_name() for fname in font_list]

    # # 显示可能的中文字体
    # chinese_fonts = []
    # for name in font_names:
    #     # 尝试识别可能的中文字体
    #     if any(keyword in name for keyword in ['Han', 'Song', 'Kai', 'Hei', 'Ming', 'Yuan', 'Gothic', 'WenQuanYi', 'Noto Sans CJK', 'Source Han']):
    #         chinese_fonts.append(name)
    # if len(chinese_fonts)>=2:
    #     plt.rcParams['font.sans-serif'] = [chinese_fonts[1]] + plt.rcParams['font.sans-serif']
    # elif len(chinese_fonts)>=1:
    #     plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # plt.rcParams['font.sans-serif'].insert(0, 'WenQuanYi Zen Hei')  # 将 SimHei 插入到 sans-serif 列表的第一位
    plt.rcParams.update({'font.size': 13})

    # 将索引转换为日期格式
    nav.index = pd.to_datetime(nav.index)
    extra_rtn_results['nav'].index = pd.to_datetime(extra_rtn_results['nav'].index)
    icresults['cumulation_rank_IC'].index = pd.to_datetime(icresults['cumulation_rank_IC'].index)

    # 创建一个 3x2 的图表布局
    fig, axs = plt.subplots(3, 2, figsize=(21, 18))  # 2 行 2 列

    # 第一组文字
    # axs[0, 0].text(0.5, 0.5,
    #             f"每组年化收益率:\n {annualized_returns}",
    #             ha='center', va='center', fontsize=24)
    # axs[0, 0].axis('off')  # 关闭坐标轴

    df_ann_return = pd.DataFrame(annualized_returns).reset_index().rename(columns={'index':'group index'})
    df_ann_return['group index'] = df_ann_return['group index'].astype(int)+1
    axs[0, 0].axis('tight')
    axs[0, 0].axis('off')
    table = axs[0, 0].table(cellText=df_ann_return.values, colLabels=df_ann_return.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1.2, 1.2)  # 调整表格大小

    # 第二组文字
    axs[0, 1].text(0.5, 0.5,
                    f"Description: {description} \n"
                    f"Average IR: {icresults['ICIR']:.4f} \n"
                    f"Average Rank(IC): {icresults['average_rank_IC']:.4f}\n"
                    f"多空收益率: {AnnualizedReturnFun(nav[-1]-1,len(nav)):.4f} \n"
                    f"换手率: {turnover:.4f}\n"
                    f"多头超额收益(年化):{extra_rtn_results['ann_rtn']:.4f} \n"
                    f"多头超额收益（年化）/波动率：{extra_rtn_results['index_sharpe']:.4f} \n"
                    f"多头超额收益(累计净值):{extra_rtn_results['nav'].iloc[-1]['extra_nav']:.4f} \n"
                    f"多头换手率: {extra_rtn_results['turnover']:.4f}",
                    ha='center', va='center', fontsize=28)
    axs[0, 1].axis('off')  # 关闭坐标轴


    # 第一个图：年化收益率柱状图
    axs[1, 0].bar(range(1, group_num + 1), annualized_returns, color='skyblue')
    axs[1, 0].set_xlabel('组别', fontsize=14)
    axs[1, 0].set_ylabel('年化收益率', fontsize=14)
    axs[1, 0].set_title("每组年化收益率", fontsize=16)
    axs[1, 0].set_xticks(range(1, group_num + 1))

    # 第二个图：超额收益的累计净值
    axs[1, 1].plot(extra_rtn_results['nav']['extra_nav'], color='orange', linewidth=2)
    axs[1, 1].set_xlabel('日期', fontsize=14)
    axs[1, 1].set_ylabel('累计净值', fontsize=14)
    axs[1, 1].set_title(
            f"多头超额净值（券池等权）（多头超额年化收益率={extra_rtn_results['ann_rtn']:.4f},多头换手率={extra_rtn_results['turnover']:.4f}",
            fontsize=16)

    # 第三个图：整体累计净值
    axs[2, 0].plot(pd.to_datetime(nav.index), nav, color='orange', label='累计净值', linewidth=2)
    axs[2, 0].set_xlabel('日期', fontsize=14)
    axs[2, 0].set_ylabel('累计净值', fontsize=14)
    axs[2, 0].set_title(f"多空净值 (NAV) Rankic ={icresults['average_rank_IC']:.4f},多空换手率={turnover:.4f}", fontsize=16)

    # 第四个图：累计 Rank_IC
    axs[2, 1].plot(icresults['cumulation_rank_IC'].cumsum(), color='orange', label='累计Rank_IC', linewidth=2)
    axs[2, 1].set_xlabel('日期', fontsize=14)
    axs[2, 1].set_ylabel('累计Rank_IC', fontsize=14)
    axs[2, 1].set_title("累积Rank_IC", fontsize=16)

    # 调整子图间距，避免标题和标签重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    # 显示图表
    if save_dir:
        plt.savefig(os.path.join(save_dir,'fig.png'))
    plt.show()
    return [icresults['average_rank_IC'], icresults['ICIR'], AnnualizedReturnFun(nav[-1]-1,len(nav)), extra_rtn_results['ann_rtn'], extra_rtn_results['index_sharpe'], extra_rtn_results['turnover']]












def factor_eval_main(args):
    factor_df, returns_df = load_dfs(args['inf_path'], args['return_path'], args['eval_start'], args['eval_end'])
    stats = plot_all(factor_df, returns_df,args['save_dir'],args['description'])
    excel_data = [args['description'], '42factor_HFQ', args['arch'], args['mode']]+stats+[f"{args['eval_start']}-{args['eval_end']}"]+[args['save_dir']]

    # 打开文件并以追加模式写入
    with open(os.path.join(args['save_dir'],'excel_data.csv'), mode='a') as file:
        writer = csv.writer(file, delimiter='\t')  # 设置制表符作为分隔符
        writer.writerow(excel_data)  # 将列表作为一行写入CSV文件

def _factor_eval_main(args):
    factor_df, returns_df = load_dfs(args)
    plot_all(factor_df, returns_df,args['save_dir'])
    

if __name__ == "__main__":
    config = {
        'train_start': '2016-01-01',
        'train_end': '2016-11-30',
        'inf_start':'2021-01-01',
        'inf_end':'2024-12-27',
        "feature_path": "factor_HFD03_HFD59_bar4_20160101_20241231.h5",
        'eval_stock_list': 'SCI500_code.xlsx',
        "label_path": "label_rq_1Dvwap_0p2class_20100101_20241231.csv.gz",
        "available_path":"allA_available_20050101_20241231.csv.xz",
        "return_path":"allA_T2_T1_vwap_rq_20100101_20231231.csv.xz",
        'save_dir':'GP-CNN',#
        'normalization': 'zscore',
    }
    _factor_eval_main(config)
