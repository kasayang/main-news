import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

class FinancialAnalyzer:
    """
    財務指標分析器，負責計算和分析公司財務數據
    """
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """
        初始化財務分析器
        
        Args:
            database_path (str): 資料庫路徑
        """
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger("FinancialAnalyzer")
        
        # 資料庫連接
        self.conn = sqlite3.connect(database_path)
    
    def get_financial_report(self, stock_id: str, report_type: str = '季報') -> pd.DataFrame:
        """
        獲取特定股票的財務報告
        
        Args:
            stock_id (str): 股票代碼
            report_type (str): 報告類型（季報/年報）
        
        Returns:
            DataFrame: 財務報告資料
        """
        query = """
        SELECT *
        FROM financial_reports
        WHERE stock_id = ?
        AND report_type = ?
        ORDER BY report_date DESC
        LIMIT 8  # 最近8個季度/年度報告
        """
        
        df = pd.read_sql(query, self.conn, params=(stock_id, report_type))
        df['report_date'] = pd.to_datetime(df['report_date'])
        
        return df
    
    def calculate_financial_ratios(self, stock_id: str, report_type: str = '季報') -> Dict:
        """
        計算財務比率
        
        Args:
            stock_id (str): 股票代碼
            report_type (str): 報告類型（季報/年報）
        
        Returns:
            Dict: 財務比率
        """
        # 獲取財務報告
        reports = self.get_financial_report(stock_id, report_type)
        
        if reports.empty:
            self.logger.warning(f"找不到 {stock_id} 的財務報告")
            return {}
        
        # 取最新一季報告
        latest_report = reports.iloc[0]
        
        # 計算財務比率
        ratios = {
            # 盈利能力指標
            'ROE': self._safe_get_value(latest_report, '股東權益報酬率'),
            'ROA': self._safe_get_value(latest_report, '資產報酬率'),
            'Net_Profit_Margin': self._safe_get_value(latest_report, '淨利率'),
            'Gross_Profit_Margin': self._safe_get_value(latest_report, '毛利率'),
            
            # 償債能力指標
            'Current_Ratio': self._safe_get_value(latest_report, '流動比率'),
            'Debt_Ratio': self._safe_get_value(latest_report, '負債比率'),
            'Debt_to_Equity': self._safe_get_value(latest_report, '負債權益比'),
            
            # 營運能力指標
            'Asset_Turnover': self._safe_get_value(latest_report, '總資產週轉率'),
            'Inventory_Turnover': self._safe_get_value(latest_report, '存貨週轉率'),
            'Receivables_Turnover': self._safe_get_value(latest_report, '應收帳款週轉率'),
            
            # 獲利能力指標
            'EPS': self._safe_get_value(latest_report, '每股盈餘'),
            'Operating_Profit_Margin': self._safe_get_value(latest_report, '營業利益率'),
            
            # 其他重要指標
            'Revenue': self._safe_get_value(latest_report, '營收'),
            'Net_Income': self._safe_get_value(latest_report, '淨利'),
            'Total_Assets': self._safe_get_value(latest_report, '總資產'),
            'Total_Equity': self._safe_get_value(latest_report, '股東權益')
        }
        
        # 計算趨勢指標
        trend_indicators = self._calculate_financial_trends(reports)
        ratios.update(trend_indicators)
        
        return ratios
    
    def _safe_get_value(self, report: pd.Series, column: str, default: float = np.nan) -> float:
        """
        安全取得報告中的值
        
        Args:
            report (Series): 財務報告
            column (str): 欄位名稱
            default (float): 默認值
        
        Returns:
            float: 欄位值
        """
        try:
            return float(report[column]) if column in report.index else default
        except (ValueError, TypeError):
            return default
    
    def _calculate_financial_trends(self, reports: pd.DataFrame) -> Dict:
        """
        計算財務趨勢指標
        
        Args:
            reports (DataFrame): 財務報告列表
        
        Returns:
            Dict: 趨勢指標
        """
        trend_indicators = {}
        
        # 需要計算趨勢的指標
        trend_columns = [
            '營收', '淨利', '每股盈餘', 
            '股東權益報酬率', '資產報酬率'
        ]
        
        for column in trend_columns:
            if column in reports.columns:
                # 移除非數值
                values = pd.to_numeric(reports[column], errors='coerce').dropna()
                
                if len(values) > 1:
                    # 計算變化率
                    change_rates = values.pct_change().dropna()
                    
                    trend_indicators[f'{column}_avg_growth'] = change_rates.mean()
                    trend_indicators[f'{column}_growth_volatility'] = change_rates.std()
                    
                    # 線性回歸斜率（趨勢）
                    x = np.arange(len(values))
                    slope, _ = np.polyfit(x, values, 1)
                    trend_indicators[f'{column}_trend_slope'] = slope
        
        return trend_indicators
    
    def compare_stocks_financial_health(self, stock_ids: List[str]) -> pd.DataFrame:
        """
        比較多支股票的財務健康狀況
        
        Args:
            stock_ids (List[str]): 股票代碼列表
        
        Returns:
            DataFrame: 股票財務比較結果
        """
        results = []
        
        for stock_id in stock_ids:
            try:
                ratios = self.calculate_financial_ratios(stock_id)
                ratios['stock_id'] = stock_id
                results.append(ratios)
            except Exception as e:
                self.logger.warning(f"計算 {stock_id} 財務指標時發生錯誤: {e}")
        
        return pd.DataFrame(results)
    
    def financial_health_score(self, stock_id: str) -> Dict:
        """
        綜合評估公司財務健康
        
        Args:
            stock_id (str): 股票代碼
        
        Returns:
            Dict: 財務健康得分
        """
        ratios = self.calculate_financial_ratios(stock_id)
        
        # 設定評分標準
        score_criteria = {
            'Profitability': {
                'ROE': (0.1, 'higher_is_better'),
                'Net_Profit_Margin': (0.05, 'higher_is_better'),
                'Operating_Profit_Margin': (0.05, 'higher_is_better')
            },
            'Solvency': {
                'Current_Ratio': (0.1, 'higher_is_better'),
                'Debt_Ratio': (0.1, 'lower_is_better'),
                'Debt_to_Equity': (0.05, 'lower_is_better')
            },
            'Operational_Efficiency': {
                'Asset_Turnover': (0.1, 'higher_is_better'),
                'Receivables_Turnover': (0.05, 'higher_is_better')
            }
        }
        
        health_score = {
            'total_score': 0,
            'category_scores': {}
        }
        
        for category, indicators in score_criteria.items():
            category_score = 0
            for indicator, (weight, direction) in indicators.items():
                # 預設值
                value = ratios.get(indicator, 0)
                
                # 不同方向的評分
                if direction == 'higher_is_better':
                    # 假設大於10%為滿分，小於0為0分
                    score = min(max(value / 0.1, 0), 1) * weight
                else:
                    # 假設低於10%為滿分，高於50%為0分
                    score = max(1 - value / 0.5, 0) * weight
                
                category_score += score
            
            health_score['category_scores'][category] = category_score
            health_score['total_score'] += category_score
        
        # 將總分映射到0-100
        health_score['total_score'] *= 100
        
        return health_score
    
    def earnings_forecast(self, stock_id: str, forecast_quarters: int = 4) -> Dict:
        """
        根據歷史財務報告預測未來收益
        
        Args:
            stock_id (str): 股票代碼
            forecast_quarters (int): 預測季度數
        
        Returns:
            Dict: 收益預測
        """
        reports = self.get_financial_report(stock_id)
        
        if len(reports) < 4:
            return {'error': '資料不足，無法進行預測'}
        
        # 計算歷史每股盈餘成長率
        eps_values = pd.to_numeric(reports['每股盈餘'], errors='coerce')
        growth_rates = eps_values.pct_change().dropna()
        
        # 平均成長率
        avg_growth_rate = growth_rates.mean()
        
        # 最近一季的每股盈餘
        latest_eps = eps_values.iloc[0]
        
        # 預測未來4個季度的每股盈餘
        forecast_eps = [latest_eps * (1 + avg_growth_rate) ** (i+1) for i in range(forecast_quarters)]
        
        return {
            'latest_eps': latest_eps,
            'avg_growth_rate': avg_growth_rate,
            'forecast_eps': forecast_eps
        }
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()

# 使用範例
def main():
    analyzer = FinancialAnalyzer()
    try:
        # 分析單支股票財務指標
        indicators = analyzer.calculate_financial_ratios('2330')
        print("台積電財務指標:")
        for key, value in indicators.items():
            print(f"{key}: {value}")
        
        # 財務健康評分
        health_score = analyzer.financial_health_score('2330')
        print("\n財務健康得分:")
        print(health_score)
        
        # 收益預測
        earnings_forecast = analyzer.earnings_forecast('2330')
        print("\n收益預測:")
        print(earnings_forecast)
        
        # 比較多支股票的財務狀況
        comparison = analyzer.compare_stocks_financial_health(['2330', '2317', '2454'])
        print("\n多股票財務比較:")
        print(comparison)
    
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()