import requests
import pandas as pd
import sqlite3
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import os

class FinancialCollector:
    """
    財務數據收集器，負責從多個來源收集公司財務報告和基本面資訊
    """
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """
        初始化財務數據收集器
        
        Args:
            database_path (str): 資料庫路徑
        """
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger("FinancialCollector")
        
        # 資料庫連接
        self.conn = sqlite3.connect(database_path)
        
        # 創建必要的資料表
        self._create_tables()
        
        # 財報 API 設定
        self.mops_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def _create_tables(self):
        """
        創建必要的資料庫表格
        """
        cursor = self.conn.cursor()
        
        # 財務報告表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_reports (
            stock_id TEXT,
            report_type TEXT,  # 季報/年報
            report_date DATE,
            revenue REAL,      # 營收
            net_profit REAL,   # 淨利
            eps REAL,          # 每股盈餘
            total_assets REAL, # 總資產
            total_liabilities REAL,  # 總負債
            equity REAL,       # 股東權益
            gross_profit_margin REAL,  # 毛利率
            operating_profit_margin REAL,  # 營業利益率
            net_profit_margin REAL,    # 淨利率
            PRIMARY KEY (stock_id, report_type, report_date)
        )
        ''')
        
        # 股東權益變動表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shareholders_equity (
            stock_id TEXT,
            report_date DATE,
            total_shareholders REAL,  # 總股東人數
            foreign_shareholders_ratio REAL,  # 外資持股比例
            investment_trust_ratio REAL,  # 投信持股比例
            PRIMARY KEY (stock_id, report_date)
        )
        ''')
        
        # 股利分配表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dividend_distribution (
            stock_id TEXT,
            year INTEGER,
            cash_dividend REAL,    # 現金股利
            stock_dividend REAL,   # 股票股利
            total_dividend REAL,   # 總股利
            dividend_yield REAL,   # 股利率
            PRIMARY KEY (stock_id, year)
        )
        ''')
        
        self.conn.commit()
    
    def fetch_mops_financial_report(self, stock_id: str, year: int, season: int) -> Dict:
        """
        從公開資訊觀測站爬取財務報告
        
        Args:
            stock_id (str): 股票代碼
            year (int): 西元年
            season (int): 季度 (1-4)
        
        Returns:
            Dict: 財務報告資料
        """
        try:
            # 公開資訊觀測站財報網址
            url = "https://mops.twse.com.tw/mops/web/t163sb05"
            
            # 請求參數
            payload = {
                'encodeURIComponent': '1',
                'step': '2',
                'firstin': '1',
                'off': '1',
                'isQuery': 'Y',
                'TYPEK': 'sii',
                'year': str(year - 1911),  # 轉換為民國年
                'season': str(season),
                'co_id': stock_id
            }
            
            response = requests.post(url, data=payload, headers=self.mops_headers)
            
            # 解析表格
            df = pd.read_html(response.text)
            
            # 處理財務資料
            financial_data = {}
            
            for table in df:
                # 轉換為字典，移除多級標題
                table.columns = table.columns.get_level_values(-1)
                table_dict = table.to_dict('records')[0]
                
                # 提取關鍵財務指標
                financial_data.update({
                    'revenue': self._parse_financial_value(table_dict.get('營業收入', 0)),
                    'net_profit': self._parse_financial_value(table_dict.get('本期淨利', 0)),
                    'eps': self._parse_financial_value(table_dict.get('基本每股盈餘', 0)),
                    'total_assets': self._parse_financial_value(table_dict.get('資產總額', 0)),
                    'total_liabilities': self._parse_financial_value(table_dict.get('負債總額', 0)),
                    'equity': self._parse_financial_value(table_dict.get('歸屬於母公司業主之權益', 0)),
                    'gross_profit_margin': self._parse_percentage(table_dict.get('營業毛利', 0)),
                    'operating_profit_margin': self._parse_percentage(table_dict.get('營業利益', 0)),
                    'net_profit_margin': self._parse_percentage(table_dict.get('本期淨利', 0))
                })
            
            return financial_data
        
        except Exception as e:
            self.logger.error(f"獲取 {stock_id} 財務報告時發生錯誤: {e}")
            return {}
    
    def _parse_financial_value(self, value: str) -> float:
        """
        解析財務數值
        
        Args:
            value (str): 原始數值字串
        
        Returns:
            float: 解析後的數值
        """
        try:
            # 移除千分位逗號，轉換為浮點數
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_percentage(self, value: str) -> float:
        """
        解析百分比
        
        Args:
            value (str): 原始百分比字串
        
        Returns:
            float: 解析後的百分比
        """
        try:
            # 移除 '%'，轉換為浮點數
            return float(str(value).replace('%', '').replace(',', '')) / 100
        except (ValueError, TypeError):
            return 0.0
    
    def batch_fetch_financial_reports(self, stock_ids: List[str], year: Optional[int] = None, 
                                      season: Optional[int] = None):
        """
        批次獲取多支股票的財務報告
        
        Args:
            stock_ids (List[str]): 股票代碼列表
            year (int, optional): 年份，默認為當前年份
            season (int, optional): 季度，默認為當前季度
        """
        # 設定默認年份和季度
        if year is None:
            year = datetime.now().year
        
        if season is None:
            season = (datetime.now().month - 1) // 3 + 1
        
        for stock_id in stock_ids:
            try:
                # 獲取財務報告
                financial_data = self.fetch_mops_financial_report(stock_id, year, season)
                
                if financial_data:
                    # 插入資料庫
                    cursor = self.conn.cursor()
                    cursor.execute('''
                    INSERT OR REPLACE INTO financial_reports 
                    (stock_id, report_type, report_date, 
                     revenue, net_profit, eps, 
                     total_assets, total_liabilities, equity, 
                     gross_profit_margin, operating_profit_margin, net_profit_margin)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        stock_id, 
                        f'{year}Q{season}', 
                        f'{year}-{season*3}',
                        financial_data.get('revenue', 0),
                        financial_data.get('net_profit', 0),
                        financial_data.get('eps', 0),
                        financial_data.get('total_assets', 0),
                        financial_data.get('total_liabilities', 0),
                        financial_data.get('equity', 0),
                        financial_data.get('gross_profit_margin', 0),
                        financial_data.get('operating_profit_margin', 0),
                        financial_data.get('net_profit_margin', 0)
                    ))
                    
                    self.conn.commit()
                    self.logger.info(f"成功獲取 {stock_id} {year}Q{season} 財務報告")
            
            except Exception as e:
                self.logger.error(f"處理 {stock_id} 財務報告時發生錯誤: {e}")
    
    def fetch_dividend_distribution(self, stock_id: str, start_year: int = 2010) -> List[Dict]:
        """
        獲取股票股利分配歷史
        
        Args:
            stock_id (str): 股票代碼
            start_year (int): 起始年份
        
        Returns:
            List[Dict]: 股利分配資料
        """
        try:
            current_year = datetime.now().year
            dividend_data = []
            
            for year in range(start_year, current_year + 1):
                # 模擬股利資料獲取（實際應替換為真實的 API 或爬蟲）
                url = f"https://example.com/dividend/{stock_id}/{year}"
                
                # 這裡應該是真實的 API 調用或爬蟲
                # 以下為模擬數據
                mock_dividend = {
                    'stock_id': stock_id,
                    'year': year,
                    'cash_dividend': 2.5,  # 每股現金股利
                    'stock_dividend': 0.0,  # 每股股票股利
                    'total_dividend': 2.5,  # 總股利
                    'dividend_yield': 0.03  # 股利率
                }
                
                dividend_data.append(mock_dividend)
                
                # 插入資料庫
                cursor = self.conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO dividend_distribution 
                (stock_id, year, cash_dividend, stock_dividend, total_dividend, dividend_yield)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    mock_dividend['stock_id'],
                    mock_dividend['year'],
                    mock_dividend['cash_dividend'],
                    mock_dividend['stock_dividend'],
                    mock_dividend['total_dividend'],
                    mock_dividend['dividend_yield']
                ))
            
            self.conn.commit()
            return dividend_data
        
        except Exception as e:
            self.logger.error(f"獲取 {stock_id} 股利分配資料時發生錯誤: {e}")
            return []
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()

# 使用範例
def main():
    collector = FinancialCollector()
    try:
        # 批次獲取財務報告
        collector.batch_fetch_financial_reports(['2330', '2317'], 2023, 4)
        
        # 獲取股利分配資料
        dividends = collector.fetch_dividend_distribution('2330')
        print("台積電股利分配:")
        for dividend in dividends:
            print(dividend)
    
    finally:
        collector.close()

if __name__ == "__main__":
    main()