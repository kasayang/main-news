# 導入所有數據收集器
from .price_collector import PriceCollector
from .news_collector import NewsCollector
from .financial_collector import FinancialCollector

# 版本資訊
__version__ = "1.0.0"

# 數據收集器工廠類
class CollectorFactory:
    """
    數據收集器工廠類，用於集中管理和創建不同類型的數據收集器
    """
    
    @staticmethod
    def create_collector(collector_type: str, **kwargs):
        """
        根據類型創建數據收集器
        
        Args:
            collector_type (str): 收集器類型
            **kwargs: 初始化參數
        
        Returns:
            數據收集器實例
        """
        collector_map = {
            'price': PriceCollector,
            'news': NewsCollector,
            'financial': FinancialCollector
        }
        
        if collector_type not in collector_map:
            raise ValueError(f"不支持的收集器類型: {collector_type}")
        
        return collector_map[collector_type](**kwargs)
    
    @staticmethod
    def list_available_collectors():
        """
        列出所有可用的數據收集器
        
        Returns:
            list: 可用收集器類型
        """
        return ['price', 'news', 'financial']

# 整合的數據收集管理器
class DataCollectionManager:
    """
    數據收集管理器，協調多個數據收集器的工作
    """
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """
        初始化數據收集管理器
        
        Args:
            database_path (str): 資料庫路徑
        """
        self.database_path = database_path
        
        # 初始化各類收集器
        self.price_collector = PriceCollector(database_path)
        self.news_collector = NewsCollector(database_path)
        self.financial_collector = FinancialCollector(database_path)
    
    def collect_stock_list(self) -> list:
        """
        獲取股票清單
        
        Returns:
            list: 股票清單
        """
        return self.price_collector.fetch_twse_stock_list()
    
    def collect_comprehensive_data(self, 
                                   stock_ids: list = None, 
                                   days: int = 365, 
                                   update_all: bool = False):
        """
        綜合收集多種類型的數據
        
        Args:
            stock_ids (list, optional): 指定股票代碼列表
            days (int): 收集最近幾天的數據
            update_all (bool): 是否更新所有可用股票
        """
        # 如果未指定股票，且需要更新全部，則獲取股票清單
        if update_all or not stock_ids:
            stock_ids = [stock['stock_id'] for stock in self.collect_stock_list()]
        
        # 限制股票數量，避免過度請求
        stock_ids = stock_ids[:100]
        
        # 收集股價數據
        self.price_collector.batch_update_stock_prices(stock_ids)
        
        # 收集新聞
        self.news_collector.collect_integrated_news(stock_ids)
        
        # 收集財務報告
        current_year = datetime.now().year
        current_season = (datetime.now().month - 1) // 3 + 1
        self.financial_collector.batch_fetch_financial_reports(stock_ids, year=current_year, season=current_season)
    
    def generate_comprehensive_report(self, stock_id: str):
        """
        為特定股票生成綜合報告
        
        Args:
            stock_id (str): 股票代碼
        
        Returns:
            dict: 綜合報告
        """
        # 獲取股價數據
        price_data = self.price_collector.fetch_stock_prices(stock_id)
        
        # 獲取財務報告
        financial_reports = self.financial_collector.fetch_mops_financial_report(stock_id, 
                                                                                datetime.now().year, 
                                                                                (datetime.now().month - 1) // 3 + 1)
        
        # 獲取相關新聞
        news_data = self.news_collector.get_news_by_stock_id(stock_id)
        
        return {
            'stock_id': stock_id,
            'price_data': price_data,
            'financial_reports': financial_reports,
            'news_data': news_data
        }
    
    def close(self):
        """
        關閉所有收集器的資料庫連接
        """
        self.price_collector.close()
        self.news_collector.close()
        self.financial_collector.close()

# 導出模組
__all__ = [
    'PriceCollector', 
    'NewsCollector', 
    'FinancialCollector',
    'CollectorFactory',
    'DataCollectionManager'
]