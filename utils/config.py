import os
import json
import yaml
from typing import Dict, Any, Optional

class ConfigManager:
    """
    系統配置管理器
    支持多種配置文件格式：JSON, YAML
    """
    
    def __init__(self, config_dir: str = 'config'):
        """
        初始化配置管理器
        
        Args:
            config_dir (str): 配置文件目錄
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        
        # 預設配置
        self.default_config = {
            'database': {
                'path': 'tw_stock_data.db',
                'type': 'sqlite'
            },
            'models': {
                'price_predictor': {
                    'days': 30,
                    'test_size': 0.2
                },
                'news_model': {
                    'sentiment_threshold': 0.2
                }
            },
            'logging': {
                'level': 'INFO',
                'dir': 'logs'
            }
        }
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """
        載入配置文件
        
        Args:
            filename (str): 配置文件名稱
        
        Returns:
            Dict: 配置內容
        """
        filepath = os.path.join(self.config_dir, filename)
        
        # 判斷文件類型
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if ext == '.json':
                    config = json.load(f)
                elif ext in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的文件類型: {ext}")
            
            # 合併預設配置和用戶配置
            merged_config = self._deep_merge(self.default_config, config)
            return merged_config
        
        except FileNotFoundError:
            print(f"配置文件 {filename} 未找到，使用預設配置")
            return self.default_config
        except json.JSONDecodeError:
            print(f"JSON配置文件 {filename} 解析錯誤")
            return self.default_config
        except yaml.YAMLError:
            print(f"YAML配置文件 {filename} 解析錯誤")
            return self.default_config
    
    def save_config(self, config: Dict[str, Any], filename: str = 'config.json') -> None:
        """
        保存配置文件
        
        Args:
            config (Dict): 配置內容
            filename (str): 保存的文件名
        """
        filepath = os.path.join(self.config_dir, filename)
        
        # 判斷文件類型
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if ext == '.json':
                    json.dump(config, f, ensure_ascii=False, indent=4)
                elif ext in ['.yaml', '.yml']:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                else:
                    raise ValueError(f"不支持的文件類型: {ext}")
            print(f"配置已成功保存到 {filepath}")
        
        except Exception as e:
            print(f"保存配置文件時發生錯誤: {e}")
    
    def get_config_value(self, config: Dict[str, Any], key_path: str, default: Optional[Any] = None) -> Any:
        """
        安全取得巢狀字典的值
        
        Args:
            config (Dict): 配置字典
            key_path (str): 以點分隔的鍵路徑
            default (Any, optional): 默認值
        
        Returns:
            Any: 配置值
        """
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        遞歸合併兩個字典
        
        Args:
            base (Dict): 基礎配置
            update (Dict): 更新的配置
        
        Returns:
            Dict: 合併後的配置
        """
        result = base.copy()
        for key, value in update.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

# 全局配置管理器實例
config_manager = ConfigManager()