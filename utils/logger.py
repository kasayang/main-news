import logging
import os
from datetime import datetime

class StockLogger:
    """股票分析系統日誌管理器"""
    
    def __init__(self, 
                 log_dir: str = 'logs', 
                 log_level: int = logging.INFO,
                 log_format: str = '%(asctime)s - %(name)s - %(levelname)s: %(message)s'):
        """
        初始化日誌管理器
        
        Args:
            log_dir (str): 日誌目錄
            log_level (int): 日誌記錄級別
            log_format (str): 日誌格式
        """
        # 確保日誌目錄存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成當天的日誌文件名
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(log_dir, f'stock_analysis_{today}.log')
        
        # 配置日誌
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # 同時輸出到控制台
            ]
        )
        
        self.logger = logging.getLogger('StockAnalysis')
    
    def get_logger(self, name: str = None):
        """
        獲取特定名稱的日誌記錄器
        
        Args:
            name (str, optional): 日誌記錄器名稱
        
        Returns:
            logging.Logger: 日誌記錄器
        """
        return logging.getLogger(name) if name else self.logger
    
    @staticmethod
    def log_error(logger, error_msg: str, exception: Exception = None):
        """
        統一的錯誤日誌記錄方法
        
        Args:
            logger (logging.Logger): 日誌記錄器
            error_msg (str): 錯誤消息
            exception (Exception, optional): 異常對象
        """
        if exception:
            logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
        else:
            logger.error(error_msg)
    
    @staticmethod
    def log_warning(logger, warning_msg: str):
        """
        統一的警告日誌記錄方法
        
        Args:
            logger (logging.Logger): 日誌記錄器
            warning_msg (str): 警告消息
        """
        logger.warning(warning_msg)
    
    @staticmethod
    def log_info(logger, info_msg: str):
        """
        統一的信息日誌記錄方法
        
        Args:
            logger (logging.Logger): 日誌記錄器
            info_msg (str): 信息消息
        """
        logger.info(info_msg)

# 創建全局日誌管理器
stock_logger = StockLogger()