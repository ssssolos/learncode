""" 在获取财务数据时，在tusahre接口遇到了问题，积分不够的尴尬局面，
于是采用akshare的本地被要几个缓存方案 """

import os
import akshare as ak
import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import pickle
from datetime import datetime

# 创建财务数据缓存目录
FIN_CACHE_DIR = "data_cache/financial"
os.makedirs(FIN_CACHE_DIR, exist_ok=True)

def get_target_stocks(clean_data, all_stocks):
    """
    从现有数据中精准提取目标股票池
    :param clean_data: 你的复权日线数据 (DataFrame)
    :param all_stocks: 全A股基础数据 (DataFrame)
    :return: 有效股票代码列表
    """
    # 1. 从clean_data获取实际使用的股票
    used_stocks = clean_data['ts_code'].unique().tolist()
    
    # 2. 从all_stocks中过滤ST/*ST
    if 'name' in all_stocks.columns:
        all_stocks['is_st'] = all_stocks['name'].str.contains('ST|\*ST', na=False)
    else:
        # 安全处理：如果all_stocks没有name列，从clean_data获取
        stock_names = clean_data[['ts_code', 'name']].drop_duplicates()
        all_stocks = pd.merge(all_stocks, stock_names, on='ts_code', how='left')
        all_stocks['is_st'] = all_stocks['name'].str.contains('ST|\*ST', na=False)
    
    # 3. 过滤退市股票 (要求delist_date为空或未来日期)
    if 'delist_date' in all_stocks.columns:
        all_stocks['delist_date'] = pd.to_datetime(all_stocks['delist_date'], errors='coerce')
        valid_mask = all_stocks['delist_date'].isna() | (all_stocks['delist_date'] > datetime.now())
    else:
        valid_mask = True  # 没有退市信息时默认全部有效
    
    # 4. 合并条件
    target_stocks = all_stocks[
        (~all_stocks['is_st']) &  # 非ST
        (all_stocks['ts_code'].isin(used_stocks)) &  # 在clean_data中出现
        valid_mask  # 未退市
    ]['ts_code'].unique().tolist()
    
    print(f"🎯 精准目标股票: {len(target_stocks)}只 (从{len(used_stocks)}只日线股票中筛选)")
    return target_stocks

def get_financial_indicators_akshare(stock_code, year):
    """
    从AKShare获取单只股票年度财务指标
    :param stock_code: 股票代码 (如 '600000.SH')
    :param year: 年份 (如 2023)
    :return: 财务指标DataFrame
    """
    cache_file = os.path.join(FIN_CACHE_DIR, f"{stock_code.replace('.','_')}_{year}.pkl")
    
    # 1. 检查缓存
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # 2. 标准化股票代码 (AKShare使用纯数字+市场后缀)
    symbol = stock_code.split('.')[0]  # 去掉.SH/.SZ
    market = 'sh' if stock_code.endswith('SH') else 'sz'
    
    try:
        # 3. 获取关键财务指标 (杜邦分析+核心比率)
        df = ak.stock_financial_analysis_indicator(
            symbol=symbol,
            market=market,
            period=f"{year}"
        )
        
        if not df.empty:
            # 4. 精选关键字段 (30个核心指标)
            core_fields = [
                '净资产收益率(%)', '总资产报酬率(%)', '销售净利率(%)', '销售毛利率(%)',
                '资产负债率(%)', '流动比率', '速动比率', '存货周转率(次)', 
                '应收账款周转率(次)', '总资产周转率(次)', '每股收益(元)', 
                '每股净资产(元)', '营业收入同比增长率(%)', '净利润同比增长率(%)',
                '经营活动现金流净额同比增长率(%)', '基本每股收益同比增长率(%)',
                '归属净利润同比增长率(%)', '扣非净利润同比增长率(%)',
                '总资产同比增长率(%)', '归属股东权益同比增长率(%)'
            ]
            
            # 5. 重命名字段 + 标准化
            rename_map = {
                '净资产收益率(%)': 'roe',
                '总资产报酬率(%)': 'roa',
                '销售净利率(%)': 'net_profit_margin',
                '销售毛利率(%)': 'gross_margin',
                '资产负债率(%)': 'debt_to_assets',
                '流动比率': 'current_ratio',
                '速动比率': 'quick_ratio',
                '每股收益(元)': 'eps',
                '每股净资产(元)': 'bps',
                '净利润同比增长率(%)': 'netprofit_yoy',
                '营业收入同比增长率(%)': 'revenue_yoy'
            }
            
            # 仅保留存在的核心字段
            available_fields = [f for f in core_fields if f in df.columns]
            if available_fields:
                df = df[available_fields].copy()
                df.rename(columns=rename_map, inplace=True)
                df['ts_code'] = stock_code
                df['report_year'] = year
                
                # 6. 保存缓存
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                
                time.sleep(1.2)  # 严格遵守AKShare限流
                return df
    
    except Exception as e:
        print(f"  ⚠️ AKShare获取失败 [{stock_code}-{year}]: {str(e)}")
    
    return pd.DataFrame()

def get_financial_backup_baostock(stock_code, year):
    """
    Baostock备用方案 (当AKShare失败时)
    """
    try:
        # 1. 登录Baostock
        bs.login()
        
        # 2. 标准化代码 (Baostock使用sh.600000格式)
        bs_code = stock_code.replace('.SH', '.sh').replace('.SZ', '.sz')
        
        # 3. 获取季度财务数据 (取Q4作为年报代理)
        rs = bs.query_performance_express_report(
            code=bs_code,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31"
        )
        
        # 4. 处理结果
        data = []
        while (rs.error_code == '0') & rs.next():
            data.append(rs.get_row_data())
        
        if 
            df = pd.DataFrame(data, columns=rs.fields)
            # 转换关键字段
            financial_df = pd.DataFrame({
                'ts_code': [stock_code],
                'report_year': [year],
                'eps': [pd.to_numeric(df['eps'].iloc[0], errors='coerce')],
                'roe': [pd.to_numeric(df['roe'].iloc[0], errors='coerce')],
                'netprofit_yoy': [pd.to_numeric(df['netProfitYoy'].iloc[0], errors='coerce')],
                'revenue_yoy': [pd.to_numeric(df['revenueYoy'].iloc[0], errors='coerce')]
            })
            return financial_df
    
    except Exception as e:
        print(f"  ⚠️ Baostock备用失败 [{stock_code}-{year}]: {str(e)}")
    
    finally:
        bs.logout()
    
    return pd.DataFrame()

def build_financial_dataset(clean_data, all_stocks, start_year=2020, end_year=2024):
    """
    构建财务数据集 (无缝对接现有clean_data)
    :param clean_ 你的复权日线数据
    :param all_stocks: 全A股基础数据
    :param start_year: 起始年份
    :param end_year: 结束年份
    :return: 财务数据DataFrame
    """
    # 1. 获取精准目标股票池
    target_stocks = get_target_stocks(clean_data, all_stocks)
    
    # 2. 生成需要获取的年份
    years = list(range(start_year, end_year + 1))
    print(f"📅 需要获取 {len(years)} 个年份: {years}")
    
    # 3. 检查已有缓存
    all_financial_data = []
    failed_records = []
    
    # 4. 遍历获取
    for stock in tqdm(target_stocks, desc="获取财务数据"):
        for year in years:
            # 检查缓存是否存在
            cache_file = os.path.join(FIN_CACHE_DIR, f"{stock.replace('.','_')}_{year}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                if not df.empty:
                    all_financial_data.append(df)
                continue
            
            # 尝试AKShare
            df = get_financial_indicators_akshare(stock, year)
            
            # 备用：当AKShare失败时尝试Baostock
            if df.empty:
                print(f"  🔁 尝试Baostock备用 [{stock}-{year}]")
                df = get_financial_backup_baostock(stock, year)
            
            # 保存结果
            if not df.empty:
                all_financial_data.append(df)
            else:
                failed_records.append((stock, year))
    
    # 5. 合并数据
    if all_financial_
        financial_df = pd.concat(all_financial_data, ignore_index=True)
        print(f"✅ 成功获取: {len(financial_df)}条财务记录")
        
        # 6. 记录失败项
        if failed_records:
            pd.DataFrame(failed_records, columns=['ts_code', 'year']).to_csv(
                os.path.join(FIN_CACHE_DIR, "failed_records.csv"), index=False
            )
            print(f"❌ 失败记录: {len(failed_records)}条 (已保存到failed_records.csv)")
        
        return financial_df
    
    raise Exception("未获取到任何财务数据！请检查网络和API状态")

def align_financial_with_daily(financial_df, clean_data):
    """
    将财务数据与日线数据对齐 (处理公告滞后性)
    :param financial_df: 财务数据
    :param clean_ 日线数据
    :return: 对齐后的DataFrame
    """
    # 1. 为财务数据添加公告日期 (简化版：年报统一为次年4月30日)
    financial_df['ann_date'] = pd.to_datetime(
        financial_df['report_year'].astype(str) + '-04-30'
    )
    
    # 2. 仅保留clean_data中存在的股票
    financial_df = financial_df[financial_df['ts_code'].isin(clean_data['ts_code'].unique())]
    
    # 3. 按股票和报告年排序
    financial_df = financial_df.sort_values(['ts_code', 'report_year'])
    
    # 4. 与日线数据合并 (关键：左连接保留所有日线)
    merged = pd.merge_asof(
        clean_data.sort_values('trade_date'),
        financial_df.sort_values('ann_date'),
        by='ts_code',
        left_on='trade_date',
        right_on='ann_date',
        direction='backward'  # 取最近的已公告财务数据
    )
    
    # 5. 前向填充财务指标 (直到新公告发布)
    financial_cols = ['roe', 'eps', 'netprofit_yoy', 'revenue_yoy', 'debt_to_assets']
    for col in financial_cols:
        if col in merged.columns:
            merged[col] = merged.groupby('ts_code')[col].ffill()
    
    print(f"📈 财务数据对齐完成! 覆盖率: {merged[financial_cols[0]].notna().mean():.2%}")
    return merged

# ============== 使用示例 ==============
if __name__ == "__main__":
    # 1. 获取财务数据 (基于你现有的clean_data和all_stocks)
    financial_data = build_financial_dataset(
        clean_data=clean_data,
        all_stocks=all_stocks,
        start_year=2020,
        end_year=2024
    )
    
    # 2. 保存财务数据
    financial_data.to_parquet("data_cache/financial_data.parquet")
    print("💾 财务数据已保存到 data_cache/financial_data.parquet")
    
    # 3. 与日线数据对齐
    final_dataset = align_financial_with_daily(financial_data, clean_data)
    
    # 4. 保存最终数据集
    final_dataset.to_parquet("data_cache/full_dataset_with_financial.parquet")
    print("🎉 最终数据集已保存! 形状:", final_dataset.shape)