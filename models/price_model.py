import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class StockPricePredictor:
    """
    股價預測模型，基於技術指標進行股價走勢預測
    """
    
    def __init__(self, database_path: str = "tw_stock_data.db", model_dir: str = "models"):
        """
        初始化股價預測模型
        
        Args:
            database_path (str): 資料庫路徑
            model_dir (str): 模型儲存目錄
        """
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger("StockPricePredictor")
        
        # 資料庫和模型路徑
        self.database_path = database_path
        self.model_dir = model_dir
        
        # 確保模型目錄存在
        os.makedirs(model_dir, exist_ok=True)
        
        # 資料庫連接
        self.conn = sqlite3.connect(database_path)
    
    def prepare_features(self, 
                         stock_id: str, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None, 
                         prediction_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        準備模型訓練特徵
        
        Args:
            stock_id (str): 股票代碼
            start_date (str, optional): 開始日期
            end_date (str, optional): 結束日期
            prediction_days (int): 預測未來幾天
        
        Returns:
            Tuple[DataFrame, Series]: 特徵和標籤
        """
        try:
            # 設定日期範圍
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365*2)).strftime('%Y-%m-%d')
            
            # 查詢股價資料
            query = f"""
            SELECT date, open, high, low, close, volume,
                   (SELECT LEAD({prediction_days}) OVER (ORDER BY date) FROM price_close p2 
                    WHERE p2.stock_id = p1.stock_id AND p2.date > p1.date) as future_close
            FROM (
                SELECT stock_id, date, open, high, low, close, volume
                FROM price_close
                WHERE stock_id = '{stock_id}'
                AND date BETWEEN '{start_date}' AND '{end_date}'
            ) p1
            ORDER BY date
            """
            
            df = pd.read_sql(query, self.conn)
            
            # 計算技術指標
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA60'] = df['close'].rolling(window=60).mean()
            
            # 計算相對強弱指數 RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 計算成交量變化
            df['volume_change'] = df['volume'].pct_change()
            
            # 計算波動率
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # 創建漲跌標籤
            df['label'] = ((df['future_close'] / df['close'] - 1) > 0.02).astype(int)
            
            # 移除包含 NaN 的列
            df.dropna(inplace=True)
            
            # 準備特徵和標籤
            features = df[['open', 'high', 'low', 'close', 'volume', 
                           'MA5', 'MA20', 'MA60', 'RSI', 
                           'volume_change', 'volatility']]
            labels = df['label']
            
            return features, labels
        
        except Exception as e:
            self.logger.error(f"準備特徵時發生錯誤: {e}")
            return None, None
    
    def train_model(self, 
                    stock_id: str, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None, 
                    prediction_days: int = 1, 
                    test_size: float = 0.2) -> Dict:
        """
        訓練股價預測模型
        
        Args:
            stock_id (str): 股票代碼
            start_date (str, optional): 起始日期
            end_date (str, optional): 結束日期
            prediction_days (int): 預測未來幾天
            test_size (float): 測試集比例
        
        Returns:
            Dict: 模型訓練結果
        """
        try:
            # 準備特徵資料
            X, y = self.prepare_features(stock_id, start_date, end_date, prediction_days)
            
            if X is None or y is None:
                self.logger.error("無法獲取特徵資料")
                return None
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=42, 
                stratify=y
            )
            
            # 構建機器學習管道
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                ))
            ])
            
            # 訓練模型
            pipeline.fit(X_train, y_train)
            
            # 評估模型
            from sklearn.metrics import (
                accuracy_score, 
                classification_report, 
                confusion_matrix
            )
            
            # 預測
            y_pred = pipeline.predict(X_test)
            
            # 計算準確率
            accuracy = accuracy_score(y_test, y_pred)
            
            # 特徵重要性
            feature_importances = pd.DataFrame({
                'feature': X.columns,
                'importance': pipeline.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 保存模型
            model_path = os.path.join(
                self.model_dir, 
                f"{stock_id}_price_pred{prediction_days}d_model.pkl"
            )
            joblib.dump(pipeline, model_path)
            
            # 返回結果
            return {
                'stock_id': stock_id,
                'model_path': model_path,
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'feature_importances': feature_importances
            }
        
        except Exception as e:
            self.logger.error(f"模型訓練時發生錯誤: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def predict_price_movement(self, 
                               stock_id: str, 
                               date: Optional[str] = None, 
                               prediction_days: int = 1) -> Dict:
        """
        預測股價走勢
        
        Args:
            stock_id (str): 股票代碼
            date (str, optional): 預測日期
            prediction_days (int): 預測未來幾天
        
        Returns:
            Dict: 預測結果
        """
        try:
            # 如果未指定日期，使用最新日期
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # 嘗試載入模型
            model_path = os.path.join(
                self.model_dir, 
                f"{stock_id}_price_pred{prediction_days}d_model.pkl"
            )
            
            if not os.path.exists(model_path):
                # 如果模型不存在，先訓練模型
                self.logger.info(f"未找到 {stock_id} 的模型，開始訓練...")
                train_result = self.train_model(stock_id, prediction_days=prediction_days)
                
                if train_result is None:
                    self.logger.error("模型訓練失敗")
                    return None
                
                model_path = train_result['model_path']
            
            # 載入模型
            model = joblib.load(model_path)
            
            # 準備特徵資料（使用較長的歷史數據）
            X, _ = self.prepare_features(
                stock_id, 
                start_date=(datetime.strptime(date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d'),
                end_date=date,
                prediction_days=prediction_days
            )
            
            # 取最新一筆資料
            latest_data = X.iloc[-1:].copy()
            
            # 預測
            prediction = model.predict(latest_data)[0]
            prediction_proba = model.predict_proba(latest_data)[0]
            
            # 查詢最新收盤價
            price_query = f"""
            SELECT close 
            FROM price_close 
            WHERE stock_id = '{stock_id}' 
            ORDER BY date DESC 
            LIMIT 1
            """
            current_price = pd.read_sql(price_query, self.conn)['close'].iloc[0]
            
            # 準備結果
            result = {
                'stock_id': stock_id,
                'prediction_date': date,
                'prediction': '上漲' if prediction == 1 else '下跌',
                'probability': prediction_proba[1] if prediction == 1 else prediction_proba[0],
                'current_price': current_price,
                'prediction_days': prediction_days
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"預測 {stock_id} 股價走勢時發生錯誤: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()

def main():
    # 創建模型實例
    predictor = StockPricePredictor()
    
    try:
        # 訓練模型
        train_result = predictor.train_model('2330')
        print("模型訓練結果:")
        print(train_result)
        
        # 預測股價走勢
        prediction = predictor.predict_price_movement('2330')
        print("\n股價走勢預測:")
        print(prediction)
    
    finally:
        predictor.close()

if __name__ == "__main__":
    main()