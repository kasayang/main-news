import requests
import pandas as pd
import sqlite3
import yfinance as yf
import twstock
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class PriceCollector:
    """
    股價數據收集器，用於從多個來源收集股票價格和基本資訊
    支援台灣證券交易所、Yahoo Finance等多個數據源
    """
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """
        初始化股價收集器
        
        Args:
            database_path (str): 資料庫路徑
        """
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger("PriceCollector")
        
        # 資料庫連接
        self.conn = sqlite3.connect(database_path)
        
        # 創建必要的資料表
        self._create_tables()
    
    def _create_tables(self):
        """
        創建必要的資料庫表格
        """
        cursor = self.conn.cursor()
        
        # 股票基本資訊表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_info (
            stock_id TEXT PRIMARY KEY,
            name TEXT,
            industry TEXT,
            market TEXT,
            listed_date DATE,
            capital REAL,
            outstanding_shares REAL
        )
        ''')
        
        # 每日股價表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_prices (
            stock_id TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (stock_id, date)
        )
        ''')
        
        # 技術指標表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS technical_indicators (
            stock_id TEXT,
            date DATE,
            MA5 REAL,
            MA20 REAL,
            MA60 REAL,
            RSI REAL,
            MACD REAL,
            KD REAL,
            PRIMARY KEY (stock_id, date)
        )
        ''')
        
        self.conn.commit()
    
    def fetch_twse_stock_list(self) -> List[Dict]:
        """
        從台灣證券交易所獲取股票清單
        
        Returns:
            List[Dict]: 股票基本資訊
        """
        try:
            # 台灣證券交易所上市公司清單 API
            url = "https://www.twse.com.tw/exchangeReport/STOCK_LIST"
            
            response = requests.get(url, params={
                "response": "json",
                "type": "ALL"
            })
            
            data = response.json()
            stocks = []
            
            for stock in data.get('data', []):
                stocks.append({
                    'stock_id': stock[0].strip(),
                    'name': stock[1].strip(),
                    'market': '上市'
                })
            
            return stocks
        
        except Exception as e:
            self.logger.error(f"獲取證交所股票清單時發生錯誤: {e}")
            return []
    
    def fetch_stock_prices(self, 
                           stock_id: str, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        從 Yahoo Finance 獲取股票歷史價格
        
        Args:
            stock_id (str): 股票代碼
            start_date (str, optional): 開始日期
            end_date (str, optional): 結束日期
        
        Returns:
            DataFrame: 股票歷史價格
        """
        try:
            # 調整股票代碼格式
            yahoo_stock_id = f"{stock_id}.TW"
            
            # 設定日期範圍
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 使用 yfinance 獲取股價
            stock = yf.Ticker(yahoo_stock_id)
            df = stock.history(start=start_date, end=end_date)
            
            # 重置索引，將日期轉為列
            df.reset_index(inplace=True)
            
            # 重命名列
            df.columns = [col.lower() for col in df.columns]
            df.rename(columns={
                'date': 'date', 
                'open': 'open', 
                'high': 'high', 
                'low': 'low', 
                'close': 'close', 
                'volume': 'volume'
            }, inplace=True)
            
            # 轉換日期格式
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            self.logger.error(f"獲取 {stock_id} 股價時發生錯誤: {e}")
            return pd.DataFrame()
    
    def save_stock_prices(self, stock_id: str, prices_df: pd.DataFrame):
        """
        將股價資料保存到資料庫
        
        Args:
            stock_id (str): 股票代碼
            prices_df (DataFrame): 股價資料
        """
        try:
            # 添加股票代碼
            prices_df['stock_id'] = stock_id
            
            # 批次插入
            prices_df.to_sql(
                'daily_prices', 
                self.conn, 
                if_exists='append', 
                index=False,
                method='multi'
            )
            
            self.logger.info(f"成功保存 {stock_id} 的 {len(prices_df)} 筆股價資料")
        
        except Exception as e:
            self.logger.error(f"保存 {stock_id} 股價資料時發生錯誤: {e}")
    
    def update_stock_info(self, stocks: List[Dict]):
        """
        更新股票基本資訊
        
        Args:
            stocks (List[Dict]): 股票基本資訊列表
        """
        try:
            cursor = self.conn.cursor()
            
            for stock in stocks:
                cursor.execute('''
                INSERT OR REPLACE INTO stock_info 
                (stock_id, name, market) 
                VALUES (?, ?, ?)
                ''', (
                    stock.get('stock_id', ''),
                    stock.get('name', ''),
                    stock.get('market', '')
                ))
            
            self.conn.commit()
            self.logger.info(f"成功更新 {len(stocks)} 支股票基本資訊")
        
        except Exception as e:
            self.logger.error(f"更新股票基本資訊時發生錯誤: {e}")
    
    def batch_update_stock_prices(self, stock_ids: List[str]):
        """
        批次更新多支股票的股價
        
        Args:
            stock_ids (List[str]): 股票代碼列表
        """
        for stock_id in stock_ids:
            try:
                # 獲取股價
                prices_df = self.fetch_stock_prices(stock_id)
                
                # 保存股價
                if not prices_df.empty:
                    self.save_stock_prices(stock_id, prices_df)
            
            except Exception as e:
                self.logger.error(f"更新 {stock_id} 股價時發生錯誤: {e}")
    
    def collect_and_update_data(self):
        """
        執行完整的數據收集和更新流程
        """
        try:
            # 獲取股票清單
            stock_list = self.fetch_twse_stock_list()
            
            # 更新股票基本資訊
            self.update_stock_info(stock_list)
            
            # 批次更新股價（限制數量，避免過度請求）
            batch_size = 50
            for i in range(0, len(stock_list), batch_size):
                batch_stocks = [stock['stock_id'] for stock in stock_list[i:i+batch_size]]
                self.batch_update_stock_prices(batch_stocks)
        
        except Exception as e:
            self.logger.error(f"數據收集過程中發生錯誤: {e}")
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()

# 使用範例
def main():
    collector = PriceCollector()
    try:
        # 執行完整的數據收集
        collector.collect_and_update_data()
        
        # 單獨獲取特定股票的股價
        prices_df = collector.fetch_stock_prices('2330')
        print("台積電最近股價:")
        print(prices_df.head())
    
    finally:
        collector.close()

if __name__ == "__main__":
    main()