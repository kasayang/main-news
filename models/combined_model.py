# models/combined_model.py
import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 導入其他模型
from models.price_model import StockPricePredictor
from models.news_model import NewsModel

class CombinedModel:
    """綜合預測模型，結合技術指標、新聞情緒和基本面資料"""
    
    def __init__(self, database_path: str = "tw_stock_data.db", model_dir: str = "models"):
        """初始化綜合預測模型
        
        Args:
            database_path (str): 資料庫路徑
            model_dir (str): 模型儲存目錄
        """
        self.database_path = database_path
        self.model_dir = model_dir
        self.conn = sqlite3.connect(database_path)
        
        # 建立模型目錄
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 設置日誌
        self.logger = logging.getLogger("CombinedModel")
        
        # 初始化子模型
        self.price_predictor = StockPricePredictor(database_path=database_path, model_dir=model_dir)
        self.news_model = NewsModel(database_path=database_path, model_dir=model_dir)
        
        # 綜合模型
        self.combined_model = None
    
    def prepare_features(self, stock_id: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None, prediction_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """準備綜合特徵資料
        
        Args:
            stock_id (str): 股票代碼
            start_date (str, optional): 起始日期
            end_date (str, optional): 結束日期
            prediction_days (int): 預測未來幾天
        
        Returns:
            tuple: (特徵資料, 標籤)
        """
        try:
            # 設定日期範圍
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if start_date is None:
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365*2)).strftime('%Y-%m-%d')
            
            # 1. 獲取價格和技術指標特徵
            price_features, price_labels = self.price_predictor.prepare_features(
                stock_id, start_date, end_date, prediction_days
            )
            
            if price_features is None or price_labels is None:
                self.logger.error(f"無法獲取 {stock_id} 的價格特徵")
                return None, None
            
            # 轉換索引為日期字串
            price_features = price_features.reset_index()
            
            # 確保日期列名稱為 'date'
            if 'date' not in price_features.columns and 'index' in price_features.columns:
                price_features = price_features.rename(columns={'index': 'date'})
            
            # 將日期轉換為字符串格式，便於後續合併
            if hasattr(price_features['date'].iloc[0], 'strftime'):
                price_features['date'] = price_features['date'].dt.strftime('%Y-%m-%d')
            
            # 2. 獲取新聞情緒特徵
            # 為每一天獲取情緒特徵
            sentiment_features = []
            dates = price_features['date'].unique()
            
            for date in dates:
                # 獲取當日之前30天的新聞情緒
                end_date_obj = datetime.strptime(date, '%Y-%m-%d')
                start_date_obj = end_date_obj - timedelta(days=30)
                
                # 查詢該日期範圍的新聞
                query = """
                SELECT n.publish_date, s.sentiment_score, r.relevance
                FROM stock_news n
                JOIN news_stock_relation r ON n.id = r.news_id
                JOIN news_sentiment s ON n.id = s.news_id
                WHERE r.stock_id = ?
                AND n.publish_date BETWEEN ? AND ?
                ORDER BY n.publish_date DESC
                """
                
                df = pd.read_sql(query, self.conn, params=(
                    stock_id, 
                    start_date_obj.strftime('%Y-%m-%d'), 
                    end_date_obj.strftime('%Y-%m-%d')
                ))
                
                # 計算情緒特徵
                if df.empty:
                    # 如果沒有新聞，設定默認值
                    feature = {
                        'date': date,
                        'news_count': 0,
                        'weighted_sentiment': 0,
                        'sentiment_volatility': 0,
                        'recent_sentiment_change': 0,
                        'sentiment_trend': 0,
                        'positive_ratio': 0,
                        'negative_ratio': 0,
                        'neutral_ratio': 0
                    }
                else:
                    # 轉換日期
                    df['publish_date'] = pd.to_datetime(df['publish_date'])
                    
                    # 計算加權情緒分數
                    df['weighted_score'] = df['sentiment_score'] * df['relevance']
                    
                    # 計算情緒特徵
                    feature = {
                        'date': date,
                        'news_count': len(df),
                        'weighted_sentiment': df['weighted_score'].mean(),
                        'sentiment_volatility': df['weighted_score'].std() if len(df) > 1 else 0
                    }
                    
                    # 計算最近變化（最近7天與之前的差異）
                    recent_days = min(7, len(df))
                    if len(df) > recent_days:
                        recent_df = df.sort_values('publish_date', ascending=False).head(recent_days)
                        earlier_df = df.sort_values('publish_date', ascending=False).iloc[recent_days:]
                        
                        feature['recent_sentiment_change'] = recent_df['weighted_score'].mean() - earlier_df['weighted_score'].mean()
                    else:
                        feature['recent_sentiment_change'] = 0
                    
                    # 計算情緒趨勢（簡單線性回歸斜率）
                    if len(df) > 1:
                        # 按日期分組，以便計算每日平均
                        daily_sentiment = df.groupby('publish_date')['weighted_score'].mean().reset_index()
                        daily_sentiment = daily_sentiment.sort_values('publish_date')
                        
                        if len(daily_sentiment) > 1:
                            x = np.arange(len(daily_sentiment))
                            y = daily_sentiment['weighted_score'].values
                            # 線性回歸
                            slope = np.polyfit(x, y, 1)[0]
                            feature['sentiment_trend'] = slope
                        else:
                            feature['sentiment_trend'] = 0
                    else:
                        feature['sentiment_trend'] = 0
                    
                    # 計算情緒分佈
                    positive_threshold = 0.2
                    negative_threshold = -0.2
                    
                    positive_count = sum(1 for score in df['sentiment_score'] if score > positive_threshold)
                    negative_count = sum(1 for score in df['sentiment_score'] if score < negative_threshold)
                    neutral_count = len(df) - positive_count - negative_count
                    
                    feature['positive_ratio'] = positive_count / len(df) if len(df) > 0 else 0
                    feature['negative_ratio'] = negative_count / len(df) if len(df) > 0 else 0
                    feature['neutral_ratio'] = neutral_count / len(df) if len(df) > 0 else 0
                
                sentiment_features.append(feature)
            
            # 轉換為DataFrame
            sentiment_df = pd.DataFrame(sentiment_features)
            
            # 3. 合併所有特徵
            # 以日期為鍵合併價格特徵和情緒特徵
            combined_features = pd.merge(price_features, sentiment_df, on='date', how='left')
            
            # 填補可能的缺失值
            combined_features = combined_features.fillna(0)
            
            # 設置索引
            combined_features = combined_features.set_index('date')
            
            # 移除非數值型特徵
            X = combined_features.select_dtypes(include=[np.number])
            
            return X, price_labels
            
        except Exception as e:
            self.logger.error(f"準備特徵時發生錯誤: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def train_model(self, stock_id: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, prediction_days: int = 1, 
                   test_size: float = 0.2) -> Dict:
        """訓練綜合預測模型
        
        Args:
            stock_id (str): 股票代碼
            start_date (str, optional): 訓練資料起始日期
            end_date (str, optional): 訓練資料結束日期
            prediction_days (int): 預測未來幾天
            test_size (float): 測試集比例
        
        Returns:
            dict: 訓練結果
        """
        try:
            # 準備特徵資料
            X, y = self.prepare_features(stock_id, start_date, end_date, prediction_days)
            
            if X is None or y is None or len(X) == 0:
                self.logger.error("無法獲取特徵資料，訓練失敗")
                return None
            
            # 檢查資料數量
            data_size = len(X)
            self.logger.info(f"訓練資料筆數: {data_size}")
            
            if data_size < 50:
                self.logger.warning(f"資料量過少 ({data_size}筆)，模型準確性可能受到影響")
                # 如果資料量過少，則使用簡單的隨機森林模型
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        min_samples_split=3,
                        random_state=42
                    ))
                ])
                
                # 分割訓練/測試集 - 時間序列分割，不進行隨機打亂
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # 訓練模型
                pipeline.fit(X_train, y_train)
                
                # 評估模型
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.logger.info(f"{stock_id} 模型訓練完成，測試集準確率: {accuracy:.4f}")
                
                # 保存模型
                model_path = os.path.join(self.model_dir, f"{stock_id}_combined_pred{prediction_days}d_model.pkl")
                joblib.dump(pipeline, model_path)
                
                # 計算特徵重要性
                if hasattr(pipeline[-1], 'feature_importances_'):
                    feature_importances = pd.DataFrame({
                        'feature': X.columns,
                        'importance': pipeline[-1].feature_importances_
                    }).sort_values('importance', ascending=False)
                else:
                    feature_importances = None
                
                # 返回結果
                result = {
                    'model': pipeline,
                    'model_path': model_path,
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'feature_importances': feature_importances,
                    'data_size': data_size
                }
                
                return result
            
            # 如果資料量足夠，使用更複雜的組合模型
            # 1. 技術指標模型
            tech_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=15, 
                random_state=42
            )
            
            # 挑選純技術指標特徵
            tech_cols = [col for col in X.columns if not (
                col.startswith('news_') or 
                col.startswith('sentiment_') or
                col.startswith('weighted_') or
                col in ['positive_ratio', 'negative_ratio', 'neutral_ratio']
            )]
            
            # 2. 新聞情緒模型
            news_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42
            )
            
            # 挑選情緒特徵
            news_cols = [col for col in X.columns if (
                col.startswith('news_') or 
                col.startswith('sentiment_') or
                col.startswith('weighted_') or
                col in ['positive_ratio', 'negative_ratio', 'neutral_ratio']
            )]
            
            # 3. 創建整合模型 - 投票分類器
            voting_model = VotingClassifier(
                estimators=[
                    ('tech', tech_model),
                    ('news', news_model)
                ],
                voting='soft',
                weights=[0.7, 0.3]  # 給技術指標更高的權重
            )
            
            # 4. 創建完整的處理管道
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('voting', voting_model)
            ])
            
            # 分割訓練/測試集 - 時間序列分割，不進行隨機打亂
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # 5. 訓練子模型
            # 技術指標模型
            tech_model.fit(X_train[tech_cols], y_train)
            
            # 新聞情緒模型 (如果有情緒特徵)
            if news_cols and len(news_cols) > 0:
                news_model.fit(X_train[news_cols], y_train)
            
            # 6. 訓練整合模型
            pipeline.fit(X_train, y_train)
            
            # 7. 評估模型
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"{stock_id} 綜合模型訓練完成，測試集準確率: {accuracy:.4f}")
            
            # 保存模型
            model_path = os.path.join(self.model_dir, f"{stock_id}_combined_pred{prediction_days}d_model.pkl")
            joblib.dump(pipeline, model_path)
            
            # 計算子模型準確率
            tech_accuracy = accuracy_score(y_test, tech_model.predict(X_test[tech_cols]))
            
            if news_cols and len(news_cols) > 0:
                news_accuracy = accuracy_score(y_test, news_model.predict(X_test[news_cols]))
            else:
                news_accuracy = 0
            
            self.logger.info(f"技術指標模型準確率: {tech_accuracy:.4f}")
            self.logger.info(f"新聞情緒模型準確率: {news_accuracy:.4f}")
            
            # 返回結果
            result = {
                'model': pipeline,
                'model_path': model_path,
                'accuracy': accuracy,
                'tech_accuracy': tech_accuracy,
                'news_accuracy': news_accuracy,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'data_size': data_size
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"訓練模型時發生錯誤: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def predict_price_movement(self, stock_id: str, date: Optional[str] = None, 
                              prediction_days: int = 1) -> Dict:
        """預測股價走勢
        
        Args:
            stock_id (str): 股票代碼
            date (str, optional): 指定日期，默認為最新日期
            prediction_days (int): 預測未來幾天
        
        Returns:
            dict: 預測結果
        """
        try:
            # 檢查綜合模型是否存在
            model_path = os.path.join(self.model_dir, f"{stock_id}_combined_pred{prediction_days}d_model.pkl")
            
            if not os.path.exists(model_path):
                # 如果綜合模型不存在，回退到使用單一模型
                self.logger.info(f"找不到 {stock_id} 的綜合模型，嘗試使用價格預測模型")
                price_prediction = self.price_predictor.predict_price_movement(stock_id, date, prediction_days)
                
                if price_prediction:
                    # 整合新聞情緒分析
                    try:
                        result = self.news_model.combine_with_price_prediction(stock_id, price_prediction)
                        result['model_type'] = "price_with_news"
                        return result
                    except:
                        # 如果新聞整合失敗，直接返回價格預測
                        self.logger.warning(f"整合新聞分析失敗，僅使用價格預測")
                        price_prediction['model_type'] = "price_only"
                        return price_prediction
                else:
                    return None
            
            # 載入綜合模型
            model = joblib.load(model_path)
            
            # 擴大時間範圍以確保有足夠資料計算特徵
            if date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                # 拉長到至少1年，確保所有特徵都能計算
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            else:
                end_date = date
                start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # 準備特徵資料
            X, _ = self.prepare_features(stock_id, start_date, end_date, prediction_days)
            
            if X is None or len(X) == 0:
                self.logger.error(f"無法獲取 {stock_id} 的特徵資料")
                return None
            
            # 只使用最新一筆資料進行預測
            latest_data = X.iloc[-1:].copy()
            
            # 進行預測
            prediction_prob = model.predict_proba(latest_data)[0]
            prediction = model.predict(latest_data)[0]
            
            # 獲取最新日期和價格
            # 查詢最新收盤價
            price_query = f"""
            SELECT date, close 
            FROM price_close 
            WHERE stock_id = '{stock_id}' 
            ORDER BY date DESC 
            LIMIT 1
            """
            price_df = pd.read_sql(price_query, self.conn)
            
            if not price_df.empty:
                latest_date = price_df['date'].iloc[0]
                current_price = price_df['close'].iloc[0]
            else:
                latest_date = datetime.now().strftime('%Y-%m-%d')
                current_price = 0
            
            # 計算目標日期
            target_date = (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=prediction_days)).strftime('%Y-%m-%d')
            
            # 整理預測結果
            result = {
                'stock_id': stock_id,
                'prediction_date': latest_date,
                'target_date': target_date,
                'prediction_days': prediction_days,
                'prediction': 'up' if prediction == 1 else 'down',
                'probability': prediction_prob[1] if prediction == 1 else prediction_prob[0],
                'current_price': current_price,
                'model_type': "combined"
            }
            
            self.logger.info(f"{stock_id} {prediction_days}天後 ({target_date}) 預測結果: {result['prediction']}, 機率: {result['probability']:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"預測 {stock_id} 漲跌時發生錯誤: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def batch_predict(self, stock_ids: List[str], prediction_days: int = 1) -> pd.DataFrame:
        """批次預測多檔股票
        
        Args:
            stock_ids (list): 股票代碼列表
            prediction_days (int): 預測未來幾天
        
        Returns:
            DataFrame: 預測結果
        """
        results = []
        
        for stock_id in stock_ids:
            result = self.predict_price_movement(stock_id, prediction_days=prediction_days)
            if result:
                results.append(result)
        
        if results:
            results_df = pd.DataFrame(results)
            return results_df
        else:
            return pd.DataFrame()
    
    def close(self):
        """關閉資料庫連接"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        
        # 關閉子模型資源
        if hasattr(self, 'price_predictor'):
            self.price_predictor.close()
        
        if hasattr(self, 'news_model'):
            self.news_model.close()