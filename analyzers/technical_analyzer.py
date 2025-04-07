import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional
import talib
import logging

class TechnicalAnalyzer:
    """
    技術指標分析器，負責計算和分析股票的技術指標
    """
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """
        初始化技術指標分析器
        
        Args:
            database_path (str): 資料庫路徑
        """
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger("TechnicalAnalyzer")
        
        # 資料庫連接
        self.conn = sqlite3.connect(database_path)
    
    def get_price_data(self, stock_id: str, days: int = 365) -> pd.DataFrame:
        """
        從資料庫獲取股票價格資料
        
        Args:
            stock_id (str): 股票代碼
            days (int): 獲取最近幾天的資料
        
        Returns:
            DataFrame: 股票價格資料
        """
        query = f"""
        SELECT date, open, high, low, close, volume
        FROM price_close
        WHERE stock_id = '{stock_id}'
        ORDER BY date DESC
        LIMIT {days}
        """
        
        df = pd.read_sql(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def calculate_indicators(self, stock_id: str, days: int = 365) -> Dict:
        """
        計算多種技術指標
        
        Args:
            stock_id (str): 股票代碼
            days (int): 計算最近幾天的指標
        
        Returns:
            Dict: 技術指標結果
        """
        # 獲取股價資料
        df = self.get_price_data(stock_id, days)
        
        # 確保資料非空
        if df.empty:
            self.logger.warning(f"股票 {stock_id} 無可用資料")
            return {}
        
        # 轉換為 NumPy 陣列
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # 計算技術指標
        indicators = {
            # 趨勢指標
            'SMA_5': talib.SMA(close, timeperiod=5)[-1],
            'SMA_20': talib.SMA(close, timeperiod=20)[-1],
            'SMA_60': talib.SMA(close, timeperiod=60)[-1],
            
            # 動量指標
            'RSI_14': talib.RSI(close, timeperiod=14)[-1],
            'MACD': talib.MACD(close)[2][-1],
            
            # 波動率指標
            'ATR_14': talib.ATR(high, low, close, timeperiod=14)[-1],
            
            # 動能指標
            'CCI_14': talib.CCI(high, low, close, timeperiod=14)[-1],
            
            # 成交量指標
            'OBV': talib.OBV(close, volume)[-1],
            
            # 布林帶指標
            'BBands_upper': talib.BBANDS(close)[0][-1],
            'BBands_middle': talib.BBANDS(close)[1][-1],
            'BBands_lower': talib.BBANDS(close)[2][-1],
        }
        
        # 計算短期和長期趨勢信號
        indicators['short_term_trend'] = self._calculate_trend(df['close'], 20)
        indicators['long_term_trend'] = self._calculate_trend(df['close'], 60)
        
        # 計算強度指標
        indicators['strength_index'] = self._calculate_strength_index(indicators)
        
        return indicators
    
    def _calculate_trend(self, prices: pd.Series, period: int) -> str:
        """
        計算股價趨勢
        
        Args:
            prices (Series): 股價序列
            period (int): 計算趨勢的週期
        
        Returns:
            str: 趨勢描述
        """
        # 取出最近 period 天的資料
        recent_prices = prices.tail(period)
        
        # 線性回歸斜率
        x = np.arange(len(recent_prices))
        slope, _ = np.polyfit(x, recent_prices, 1)
        
        # 判斷趨勢
        if slope > 0.001:
            return 'up'
        elif slope < -0.001:
            return 'down'
        else:
            return 'neutral'
    
    def _calculate_strength_index(self, indicators: Dict) -> float:
        """
        計算股票綜合強度指標
        
        Args:
            indicators (Dict): 技術指標字典
        
        Returns:
            float: 強度指數
        """
        # 權重可以根據實際研究調整
        weights = {
            'RSI_14': 0.2,
            'MACD': 0.2,
            'CCI_14': 0.15,
            'short_term_trend': 0.2,
            'long_term_trend': 0.25
        }
        
        # 將趨勢轉換為數值
        trend_map = {'up': 1, 'neutral': 0, 'down': -1}
        
        # 計算強度指數
        strength = 0
        strength += weights['RSI_14'] * (indicators['RSI_14'] / 100)
        strength += weights['MACD'] * (1 if indicators['MACD'] > 0 else -1)
        strength += weights['CCI_14'] * (indicators['CCI_14'] / 100)
        strength += weights['short_term_trend'] * trend_map[indicators['short_term_trend']]
        strength += weights['long_term_trend'] * trend_map[indicators['long_term_trend']]
        
        return (strength + 1) / 2  # 將結果映射到 0-1 範圍
    
    def generate_trading_signal(self, stock_id: str, days: int = 365) -> Dict:
        """
        生成交易信號
        
        Args:
            stock_id (str): 股票代碼
            days (int): 分析最近幾天的資料
        
        Returns:
            Dict: 交易信號
        """
        try:
            # 獲取技術指標
            indicators = self.calculate_indicators(stock_id, days)
            
            # 生成交易信號邏輯
            signal = {
                'stock_id': stock_id,
                'buy_signal': False,
                'sell_signal': False,
                'signal_strength': 0,
                'signal_description': ''
            }
            
            # 買入信號標準
            if (indicators.get('RSI_14', 0) < 30 and  # RSI 低於30
                indicators.get('MACD', 0) > 0 and     # MACD 為正
                indicators.get('short_term_trend') == 'up'):  # 短期趨勢向上
                signal['buy_signal'] = True
                signal['signal_strength'] = indicators.get('strength_index', 0)
                signal['signal_description'] = '低估且趨勢向上'
            
            # 賣出信號標準
            elif (indicators.get('RSI_14', 0) > 70 and  # RSI 高於70
                  indicators.get('MACD', 0) < 0 and     # MACD 為負
                  indicators.get('short_term_trend') == 'down'):  # 短期趨勢向下
                signal['sell_signal'] = True
                signal['signal_strength'] = 1 - indicators.get('strength_index', 0)
                signal['signal_description'] = '高估且趨勢向下'
            
            return signal
        
        except Exception as e:
            self.logger.error(f"生成 {stock_id} 交易信號時發生錯誤: {e}")
            return {}
    
    def compare_stocks(self, stock_ids: List[str], days: int = 365) -> pd.DataFrame:
        """
        比較多支股票的技術指標
        
        Args:
            stock_ids (List[str]): 股票代碼列表
            days (int): 分析最近幾天的資料
        
        Returns:
            DataFrame: 股票技術指標比較結果
        """
        results = []
        
        for stock_id in stock_ids:
            try:
                indicators = self.calculate_indicators(stock_id, days)
                signal = self.generate_trading_signal(stock_id, days)
                
                result = {
                    'stock_id': stock_id,
                    **indicators,
                    **signal
                }
                results.append(result)
            except Exception as e:
                self.logger.warning(f"計算 {stock_id} 技術指標時發生錯誤: {e}")
        
        return pd.DataFrame(results)
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()

# 使用範例
def main():
    analyzer = TechnicalAnalyzer()
    try:
        # 分析單支股票
        indicators = analyzer.calculate_indicators('2330')
        print("台積電技術指標:")
        for key, value in indicators.items():
            print(f"{key}: {value}")
        
        # 生成交易信號
        signal = analyzer.generate_trading_signal('2330')
        print("\n交易信號:")
        print(signal)
        
        # 比較多支股票
        comparison = analyzer.compare_stocks(['2330', '2317', '2454'])
        print("\n股票技術指標比較:")
        print(comparison)
    
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()