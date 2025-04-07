import os
import sys
import argparse
import logging
from datetime import datetime

# 將專案根目錄添加到 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 導入各模組
from data_collectors import DataCollectionManager
from models import ModelFactory
from analyzers import AnalyzerFactory
from utils.logger import StockLogger
from utils.config import ConfigManager

class StockAnalysisApp:
    """
    股票分析系統主應用程式
    """
    
    def __init__(self, config_path: str = 'config/app_config.json'):
        """
        初始化股票分析系統
        
        Args:
            config_path (str): 配置文件路徑
        """
        # 初始化日誌
        self.logger = StockLogger().get_logger('StockAnalysisApp')
        
        # 載入配置
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)
        
        # 初始化數據收集管理器
        self.data_collector = DataCollectionManager(
            database_path=self.config.get('database', {}).get('path', 'tw_stock_data.db')
        )
        
        # 初始化模型工廠
        self.model_factory = ModelFactory()
        
        # 初始化分析器工廠
        self.analyzer_factory = AnalyzerFactory()
    
    def update_data(self, update_all: bool = False, stock_ids: list = None):
        """
        更新股票數據
        
        Args:
            update_all (bool): 是否更新所有可用股票
            stock_ids (list, optional): 指定股票代碼列表
        """
        try:
            self.logger.info("開始更新股票數據...")
            
            # 如果未指定股票，獲取股票清單
            if update_all or not stock_ids:
                stock_list = self.data_collector.collect_stock_list()
                stock_ids = [stock['stock_id'] for stock in stock_list]
            
            # 綜合收集數據
            self.data_collector.collect_comprehensive_data(
                stock_ids=stock_ids, 
                update_all=update_all
            )
            
            self.logger.info("股票數據更新完成")
        
        except Exception as e:
            self.logger.error(f"更新股票數據時發生錯誤: {e}")
    
    def analyze_stock(self, stock_id: str):
        """
        分析特定股票
        
        Args:
            stock_id (str): 股票代碼
        
        Returns:
            dict: 分析結果
        """
        try:
            # 生成綜合報告
            comprehensive_report = self.data_collector.generate_comprehensive_report(stock_id)
            
            # 使用不同分析器進行分析
            technical_analyzer = self.analyzer_factory.create_analyzer('technical')
            news_analyzer = self.analyzer_factory.create_analyzer('news')
            financial_analyzer = self.analyzer_factory.create_analyzer('financial')
            
            # 計算技術指標
            technical_indicators = technical_analyzer.calculate_indicators(stock_id)
            
            # 分析新聞情緒
            news_sentiment = news_analyzer.get_sentiment_summary(stock_id)
            
            # 財務分析
            financial_ratios = financial_analyzer.calculate_financial_ratios(stock_id)
            
            # 整合分析結果
            analysis_result = {
                'stock_id': stock_id,
                'comprehensive_report': comprehensive_report,
                'technical_indicators': technical_indicators,
                'news_sentiment': news_sentiment,
                'financial_ratios': financial_ratios
            }
            
            return analysis_result
        
        except Exception as e:
            self.logger.error(f"分析股票 {stock_id} 時發生錯誤: {e}")
            return None
    
    def train_prediction_model(self, stock_id: str):
        """
        訓練特定股票的預測模型
        
        Args:
            stock_id (str): 股票代碼
        
        Returns:
            dict: 模型訓練結果
        """
        try:
            # 創建綜合預測模型
            combined_model = self.model_factory.create_model('combined')
            
            # 訓練模型
            model_result = combined_model.train_model(stock_id)
            
            return model_result
        
        except Exception as e:
            self.logger.error(f"訓練 {stock_id} 預測模型時發生錯誤: {e}")
            return None
    
    def predict_stock_movement(self, stock_id: str, days: int = 1):
        """
        預測股票走勢
        
        Args:
            stock_id (str): 股票代碼
            days (int): 預測天數
        
        Returns:
            dict: 預測結果
        """
        try:
            # 創建綜合預測模型
            combined_model = self.model_factory.create_model('combined')
            
            # 進行預測
            prediction_result = combined_model.predict_price_movement(
                stock_id, 
                datetime.now().strftime('%Y-%m-%d'), 
                days
            )
            
            return prediction_result
        
        except Exception as e:
            self.logger.error(f"預測 {stock_id} 股價走勢時發生錯誤: {e}")
            return None
    
    def close(self):
        """
        關閉所有資源
        """
        self.data_collector.close()

def main():
    """
    主程式入口
    """
    # 設置命令行參數解析
    parser = argparse.ArgumentParser(description='股票分析系統')
    parser.add_argument('--update', action='store_true', help='更新股票數據')
    parser.add_argument('--analyze', type=str, help='分析指定股票')
    parser.add_argument('--predict', type=str, help='預測指定股票走勢')
    parser.add_argument('--train', type=str, help='訓練指定股票的預測模型')
    
    # 解析參數
    args = parser.parse_args()
    
    # 創建應用程式實例
    app = StockAnalysisApp()
    
    try:
        # 根據參數執行相應操作
        if args.update:
            app.update_data(update_all=True)
        
        if args.analyze:
            result = app.analyze_stock(args.analyze)
            print(result)
        
        if args.predict:
            result = app.predict_stock_movement(args.predict)
            print(result)
        
        if args.train:
            result = app.train_prediction_model(args.train)
            print(result)
    
    except Exception as e:
        print(f"發生錯誤: {e}")
    
    finally:
        app.close()

if __name__ == "__main__":
    main()