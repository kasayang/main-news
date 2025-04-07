# 導入所有模型
from .price_model import StockPricePredictor
from .news_model import NewsModel
from .combined_model import CombinedModel

# 版本資訊
__version__ = "1.0.0"

# 模型工廠類
class ModelFactory:
    """
    模型工廠類，用於集中管理和創建不同類型的預測模型
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        """
        根據類型創建預測模型
        
        Args:
            model_type (str): 模型類型
            **kwargs: 初始化參數
        
        Returns:
            預測模型實例
        """
        model_map = {
            'price': StockPricePredictor,
            'news': NewsModel,
            'combined': CombinedModel
        }
        
        if model_type not in model_map:
            raise ValueError(f"不支持的模型類型: {model_type}")
        
        return model_map[model_type](**kwargs)
    
    @staticmethod
    def list_available_models():
        """
        列出所有可用模型
        
        Returns:
            list: 可用模型類型
        """
        return ['price', 'news', 'combined']

# 導出模型
__all__ = [
    'StockPricePredictor', 
    'NewsModel', 
    'CombinedModel', 
    'ModelFactory'
]