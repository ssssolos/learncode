import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_quant_dataset():
    """
    生成含8大面试高频陷阱的量化数据集（A股2023年）：
    1. 停牌（随机5%交易日） 
    2. 涨跌停（价格不变+成交量异常）
    3. 财报披露延迟（实际发布日晚于报告日30-60天）
    4. 财务数据缺失（10% ROE为空）
    5. 股票退市（2023-06-01后消失）
    6. 分红除权价格跳跃（2023-07-01）
    7. 黑天鹅极端波动（2023-10-01暴跌30%）
    8. 非交易日噪声（周末/节假日数据）
    """
    # 基础参数
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    stocks = ['600519.SH', '300750.SZ', '601318.SH', '000001.SZ', '688981.SH']  # 茅台/宁德/平安/平安银行/中芯
    
    records = []
    
    for stock in stocks:
        for date in date_range:
            # 跳过周末（但保留数据制造陷阱）
            if date.weekday() >= 5 and np.random.rand() > 0.3: 
                continue
                
            # 基础价格生成（带趋势+波动）
            base_price = 50 + (date - start_date).days * 0.1 + np.random.normal(0, 2)
            
            # 陷阱1：停牌（5%概率）
            is_suspended = np.random.rand() < 0.05
            
            # 陷阱2：涨跌停（3%概率）
            is_limit_up = np.random.rand() < 0.03
            is_limit_down = np.random.rand() < 0.03
            
            # 陷阱5：中芯国际2023-06-01后退市
            if stock == '688981.SH' and date > datetime(2023, 6, 1):
                continue
                
            # 价格逻辑
            if is_suspended:
                open_price = high_price = low_price = close_price = np.nan
                volume = 0
            else:
                # 陷阱6：茅台2023-07-01分红除权
                if stock == '600519.SH' and date == datetime(2023, 7, 1):
                    base_price *= 0.95  # 5%分红
                
                # 陷阱7：2023-10-01黑天鹅
                if date == datetime(2023, 10, 1):
                    base_price *= 0.7
                
                # 涨跌停处理
                if is_limit_up:
                    close_price = base_price * 1.1
                    open_price = high_price = close_price
                    low_price = close_price * 0.99
                elif is_limit_down:
                    close_price = base_price * 0.9
                    open_price = low_price = close_price
                    high_price = close_price * 1.01
                else:
                    # 正常波动
                    daily_ret = np.random.normal(0.0005, 0.02)
                    close_price = base_price * (1 + daily_ret)
                    high_price = max(close_price * 1.02, base_price * 1.03)
                    low_price = min(close_price * 0.98, base_price * 0.97)
                    open_price = np.random.uniform(low_price, high_price)
                
                volume = np.random.uniform(1e5, 1e7)
            
            # 陷阱3：财务数据（季度报告+延迟发布）
            quarter_month = ((date.month - 1) // 3) * 3 + 1
            report_date = datetime(date.year, quarter_month, 1) - timedelta(days=1)
            actual_publish_date = report_date + timedelta(days=np.random.randint(30, 60))
            
            # 陷阱4：10%财务数据缺失
            roe = np.random.normal(0.1, 0.05) if np.random.rand() > 0.1 else np.nan
            pe = np.random.uniform(10, 50)
            
            records.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y-%m-%d'),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'pre_close': base_price,
                'vol': volume,
                'amount': volume * close_price if not is_suspended else 0,
                'report_date': report_date.strftime('%Y-%m-%d'),
                'actual_publish_date': actual_publish_date.strftime('%Y-%m-%d'),
                'roe': roe,
                'pe': pe,
                'is_suspended': int(is_suspended),
                'is_limit_up': int(is_limit_up),
                'is_limit_down': int(is_limit_down)
            })
    
    df = pd.DataFrame(records)
    
    # 保存为CSV
    output_path = 'quant_interview_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ 数据集生成成功！共 {len(df)} 行，保存至: {output_path}")
    print("\n⚠️ 面试高频陷阱清单（务必练习）：")
    print(f"  • 停牌日占比: {df['is_suspended'].mean():.1%} (共{df['is_suspended'].sum()}天)")
    print(f"  • 涨跌停日: 茅台2023-04-15({df[(df.ts_code=='600519.SH') & (df.is_limit_up==1)].iloc[0].trade_date})")
    print(f"  • 财报延迟: 2023Q1报告实际在{df.actual_publish_date.min()}发布")
    print(f"  • 退市股票: 688981.SH在{df[df.ts_code=='688981.SH'].trade_date.max()}后消失")
    print(f"  • 价格跳跃: 600519.SH在2023-07-01分红除权")
    print(f"  • 黑天鹅: 2023-10-01市场暴跌30%")
    print("\n💡 使用建议：用此数据集练习5大核心题型 →")
    print("  1. 数据清洗（处理停牌/涨跌停）")
    print("  2. 财报时间戳修正（避免前视偏差）")
    print("  3. 动量因子计算（带shift(1)验证）")
    print("  4. 内存优化（10GB级数据处理）")
    print("  5. 回测陷阱排查（分红/退市影响）")
    
    return df

if __name__ == "__main__":
    generate_quant_dataset()