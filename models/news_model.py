# models/news_model.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

class NewsModel:
    """新聞情緒分析模型"""
    
    def __init__(self, database_path: str = "tw_stock_data.db", model_dir: str = "models"):
        """初始化
        
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
        
        # 情緒分析模型
        self.sentiment_model = None
        
        # 設置日誌
        self.logger = logging.getLogger("NewsModel")
    
    def get_sentiment_features(self, stock_id: str, 
                              days: int = 30, 
                              include_market: bool = True) -> Dict:
        """獲取股票的新聞情緒特徵
        
        Args:
            stock_id (str): 股票代碼
            days (int): 考慮最近幾天的新聞
            include_market (bool): 是否包含大盤新聞
        
        Returns:
            dict: 特徵字典
        """
        # 計算日期範圍
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 獲取股票相關新聞
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
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        ))
        
        # 如果需要，添加大盤新聞
        if include_market:
            market_indices = ['0000', 'TSE', 'OTC']  # 大盤指標
            for idx in market_indices:
                market_query = """
                SELECT n.publish_date, s.sentiment_score, r.relevance
                FROM stock_news n
                JOIN news_stock_relation r ON n.id = r.news_id
                JOIN news_sentiment s ON n.id = s.news_id
                WHERE r.stock_id = ?
                AND n.publish_date BETWEEN ? AND ?
                ORDER BY n.publish_date DESC
                """
                
                market_df = pd.read_sql(market_query, self.conn, params=(
                    idx, 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d')
                ))
                
                # 降低大盤新聞的權重
                market_df['relevance'] = market_df['relevance'] * 0.5
                
                df = pd.concat([df, market_df])
        
        # 如果沒有新聞，返回默認特徵
        if df.empty:
            return {
                'news_count': 0,
                'recent_sentiment': 0,
                'weighted_sentiment': 0,
                'sentiment_volatility': 0,
                'recent_sentiment_change': 0,
                'sentiment_trend': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0,
                'sentiment_sum': 0
            }
        
        # 轉換日期列
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        
        # 計算加權情緒分數
        df['weighted_score'] = df['sentiment_score'] * df['relevance']
        
        # 按日期分組並計算每日加權平均分數
        daily_sentiment = df.groupby('publish_date')['weighted_score'].mean().reset_index()
        daily_sentiment = daily_sentiment.sort_values('publish_date')
        
        # 計算特徵
        features = {}
        
        # 新聞數量
        features['news_count'] = len(df)
        
        # 最近情緒分數 (最近5天)
        recent_days = 5
        recent_df = df[df['publish_date'] >= (end_date - timedelta(days=recent_days))]
        features['recent_sentiment'] = recent_df['weighted_score'].mean() if not recent_df.empty else 0
        
        # 加權平均情緒分數
        features['weighted_sentiment'] = df['weighted_score'].mean()
        
        # 情緒波動性 (標準差)
        features['sentiment_volatility'] = daily_sentiment['weighted_score'].std() if len(daily_sentiment) > 1 else 0
        
        # 最近情緒變化 (最近5天與之前的差異)
        if len(daily_sentiment) > recent_days:
            recent_sentiment = daily_sentiment.iloc[-recent_days:]['weighted_score'].mean()
            earlier_sentiment = daily_sentiment.iloc[:-recent_days]['weighted_score'].mean()
            features['recent_sentiment_change'] = recent_sentiment - earlier_sentiment
        else:
            features['recent_sentiment_change'] = 0
        
        # 情緒趨勢 (簡單線性回歸斜率)
        if len(daily_sentiment) > 1:
            x = np.arange(len(daily_sentiment))
            y = daily_sentiment['weighted_score'].values
            # 線性回歸
            slope = np.polyfit(x, y, 1)[0]
            features['sentiment_trend'] = slope
        else:
            features['sentiment_trend'] = 0
        
        # 計算正面、負面、中性新聞比例
        positive_threshold = 0.2
        negative_threshold = -0.2
        
        positive_count = sum(1 for score in df['sentiment_score'] if score > positive_threshold)
        negative_count = sum(1 for score in df['sentiment_score'] if score < negative_threshold)
        neutral_count = features['news_count'] - positive_count - negative_count
        
        features['positive_ratio'] = positive_count / features['news_count'] if features['news_count'] > 0 else 0
        features['negative_ratio'] = negative_count / features['news_count'] if features['news_count'] > 0 else 0
        features['neutral_ratio'] = neutral_count / features['news_count'] if features['news_count'] > 0 else 0
        
        # 情緒總和
        features['sentiment_sum'] = df['weighted_score'].sum()
        
        return features
    
    def predict_price_movement_from_news(self, stock_id: str, days: int = 7) -> Tuple[float, float]:
        """根據新聞情緒預測股價走勢
        
        Args:
            stock_id (str): 股票代碼
            days (int): 預測未來幾天
        
        Returns:
            tuple: (預測方向, 信心分數) - 方向：1表示上漲，-1表示下跌；信心分數：0-1
        """
        # 獲取情緒特徵
        features = self.get_sentiment_features(stock_id, days=30)
        
        # 檢查是否有訓練好的模型
        model_path = os.path.join(self.model_dir, "news_sentiment_model.pkl")
        if os.path.exists(model_path):
            try:
                # 載入模型
                model = joblib.load(model_path)
                
                # 準備特徵
                feature_vector = np.array([
                    features['weighted_sentiment'],
                    features['recent_sentiment_change'],
                    features['sentiment_trend'],
                    features['sentiment_volatility'],
                    features['positive_ratio'],
                    features['negative_ratio']
                ]).reshape(1, -1)
                
                # 預測
                prediction_proba = model.predict_proba(feature_vector)[0]
                prediction = model.predict(feature_vector)[0]
                
                # 方向（1 表示上漲，-1 表示下跌）
                direction = 1 if prediction == 1 else -1
                
                # 信心分數
                confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
                
                return direction, confidence
            
            except Exception as e:
                self.logger.error(f"使用模型預測時發生錯誤: {e}")
        
        # 若沒有模型或預測失敗，使用簡單規則
        sentiment_score = features['weighted_sentiment']
        recent_change = features['recent_sentiment_change']
        trend = features['sentiment_trend']
        
        # 綜合考慮當前情緒、最近變化與趨勢
        prediction_score = sentiment_score * 0.4 + recent_change * 0.3 + trend * 0.3
        
        # 計算信心分數
        confidence = min(abs(prediction_score) * 2, 1.0)
        
        # 預測方向
        direction = 1 if prediction_score > 0 else -1
        
        return direction, confidence
    
    def train_sentiment_model(self, days: int = 365) -> None:
        """訓練情緒分析模型
        
        使用歷史股價變動與新聞情緒之間的關係訓練模型
        
        Args:
            days (int): 使用過去幾天的資料
        """
        try:
            # 查詢所有股票的新聞情緒資料和隔日股價變動
            query = """
            WITH StockReturn AS (
                SELECT 
                    date, 
                    stock_id, 
                    close,
                    LEAD(close) OVER (PARTITION BY stock_id ORDER BY date) / close - 1 AS next_day_return
                FROM price_close
                WHERE date >= date('now', '-{} days')
            )
            SELECT 
                n.stock_id,
                n.publish_date,
                r.next_day_return,
                s.sentiment_score,
                s.positive_score,
                s.negative_score,
                s.neutral_score,
                nr.relevance
            FROM stock_news n
            JOIN news_stock_relation nr ON n.id = nr.news_id
            JOIN news_sentiment s ON n.id = s.news_id
            LEFT JOIN StockReturn r ON r.stock_id = nr.stock_id AND r.date = n.publish_date
            WHERE n.publish_date >= date('now', '-{} days')
            AND r.next_day_return IS NOT NULL
            """.format(days, days)
            
            df = pd.read_sql(query, self.conn)
            
            if df.empty:
                self.logger.warning("沒有足夠的資料來訓練模型")
                return
            
            # 按股票分組，計算每個股票的情緒特徵
            groups = df.groupby(['stock_id', 'publish_date'])
            
            # 準備訓練資料
            X_data = []
            y_data = []
            
            for (stock_id, date), group in groups:
                # 計算情緒特徵
                weighted_sentiment = (group['sentiment_score'] * group['relevance']).mean()
                positive_ratio = group['positive_score'].mean()
                negative_ratio = group['negative_score'].mean()
                
                # 獲取隔日報酬率
                next_day_return = group['next_day_return'].iloc[0]
                
                # 添加特徵向量
                X_data.append([
                    weighted_sentiment,
                    positive_ratio,
                    negative_ratio,
                    len(group)  # 新聞數量
                ])
                
                # 添加標籤 (1 表示上漲，0 表示下跌)
                y_data.append(1 if next_day_return > 0 else 0)
            
            # 轉換為Numpy陣列
            X = np.array(X_data)
            y = np.array(y_data)
            
            # 標準化特徵
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 訓練模型
            model = LogisticRegression(random_state=42, class_weight='balanced')
            model.fit(X_scaled, y)
            
            # 評估模型
            accuracy = model.score(X_scaled, y)
            self.logger.info(f"模型準確率: {accuracy:.4f}")
            
            # 保存模型
            model_data = {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'features': ['weighted_sentiment', 'positive_ratio', 'negative_ratio', 'news_count']
            }
            
            model_path = os.path.join(self.model_dir, "news_sentiment_model.pkl")
            joblib.dump(model_data, model_path)
            self.logger.info(f"模型已保存至: {model_path}")
            
            # 載入模型
            self.sentiment_model = model_data
        
        except Exception as e:
            self.logger.error(f"訓練情緒分析模型時發生錯誤: {e}")
    
    def combine_with_price_prediction(self, stock_id: str, price_prediction: Dict, 
                                     news_weight: float = 0.3) -> Dict:
        """將新聞預測與價格預測結合
        
        Args:
            stock_id (str): 股票代碼
            price_prediction (dict): 價格預測結果
            news_weight (float): 新聞在最終預測中的權重
        
        Returns:
            dict: 結合後的預測結果
        """
        # 從價格預測獲取數據
        price_direction = 1 if price_prediction['prediction'] == 'up' else -1
        price_probability = price_prediction['probability']
        
        # 獲取新聞預測
        news_direction, news_confidence = self.predict_price_movement_from_news(stock_id)
        
        # 計算綜合分數 (範圍：-1 到 1)
        combined_score = price_direction * price_probability * (1 - news_weight) + \
                         news_direction * news_confidence * news_weight
        
        # 更新預測結果
        result = price_prediction.copy()
        result['prediction'] = 'up' if combined_score > 0 else 'down'
        result['probability'] = abs(combined_score)
        result['news_direction'] = 'up' if news_direction > 0 else 'down'
        result['news_confidence'] = news_confidence
        result['combined_score'] = combined_score
        
        return result
    
    def get_related_news(self, stock_id: str, days: int = 7) -> List[Dict]:
        """獲取股票相關新聞
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近幾天
        
        Returns:
            list: 新聞列表
        """
        query = """
        SELECT n.id, n.title, n.content, n.url, n.publish_date, n.source,
               r.relevance, s.sentiment_score, s.keywords, s.summary
        FROM stock_news n
        JOIN news_stock_relation r ON n.id = r.news_id
        LEFT JOIN news_sentiment s ON n.id = s.news_id
        WHERE r.stock_id = ?
        AND n.publish_date >= date('now', '-{} days')
        ORDER BY n.publish_date DESC, r.relevance DESC
        LIMIT 20
        """.format(days)
        
        df = pd.read_sql(query, self.conn, params=(stock_id,))
        
        # 轉換為字典列表
        news_list = []
        for _, row in df.iterrows():
            news = {
                'id': row['id'],
                'title': row['title'],
                'url': row['url'],
                'publish_date': row['publish_date'],
                'source': row['source'],
                'relevance': row['relevance'],
                'sentiment_score': row['sentiment_score'],
                'summary': row['summary'],
                'keywords': row['keywords'].split(',') if pd.notna(row['keywords']) and row['keywords'] else []
            }
            news_list.append(news)
        
        return news_list
    
    def analyze_news_content(self, news_id: int, title: str, content: str) -> Dict:
        """分析新聞內容，提取關鍵信息
        
        Args:
            news_id (int): 新聞ID
            title (str): 新聞標題
            content (str): 新聞內容
        
        Returns:
            dict: 分析結果
        """
        # 合併標題和內容，標題有更高的權重
        text = title + " " + title + " " + content
        
        # 提取數字和百分比
        numbers = re.findall(r'(\d+\.?\d*)', text)
        percentages = re.findall(r'(\d+\.?\d*%)', text)
        
        # 尋找金額相關信息
        amount_patterns = [
            r'(\d+\.?\d*)\s*億',
            r'(\d+\.?\d*)\s*仟萬',
            r'(\d+\.?\d*)\s*千萬',
            r'(\d+\.?\d*)\s*萬',
            r'(\d+\.?\d*)\s*元'
        ]
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            amounts.extend(matches)
        
        # 尋找時間相關信息
        time_matches = re.findall(r'(\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}|\d{1,2}/\d{1,2})', text)
        
        # 尋找情緒相關詞彙
        positive_words = ["上漲", "成長", "獲利", "增加", "好", "看好", "強", "正面", "樂觀", "利多"]
        negative_words = ["下跌", "虧損", "衰退", "減少", "差", "看空", "弱", "負面", "悲觀", "利空"]
        
        positive_matches = [word for word in positive_words if word in text]
        negative_matches = [word for word in negative_words if word in text]
        
        # 提取可能的事件類型
        event_types = []
        event_patterns = {
            "財報": ["財報", "營收", "EPS", "獲利", "淨利", "毛利"],
            "董事會": ["董事會", "股東會", "股息", "董事", "監察人"],
            "人事": ["總經理", "董事長", "執行長", "CEO", "人事"],
            "產品": ["新產品", "發表", "推出", "上市"],
            "合作": ["合作", "簽約", "協議", "併購", "收購"],
            "政策": ["政策", "法規", "監管", "政府", "補助"],
            "產能": ["產能", "擴廠", "產量", "擴產"],
            "訂單": ["訂單", "出貨量", "出貨", "銷售"],
            "展望": ["展望", "預期", "前景", "看好", "看淡", "預估"],
            "投資": ["投資", "增資", "募資", "減資"],
            "法說會": ["法說會", "說明會", "法人說明會"]
        }
        
        for event_type, keywords in event_patterns.items():
            if any(keyword in text for keyword in keywords):
                event_types.append(event_type)
        
        return {
            'news_id': news_id,
            'numbers': numbers,
            'percentages': percentages,
            'amounts': amounts,
            'times': time_matches,
            'positive_words': positive_matches,
            'negative_words': negative_matches,
            'event_types': event_types
        }
    
    def get_sentiment_summary(self, stock_id: str, days: int = 30) -> Dict:
        """獲取特定股票的新聞情緒摘要
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近幾天的新聞
        
        Returns:
            dict: 情緒摘要
        """
        news_df = self.get_related_news(stock_id, days)
        
        if not news_df:
            return {
                'stock_id': stock_id,
                'news_count': 0,
                'avg_sentiment': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0
            }
        
        # 轉換為 DataFrame
        df = pd.DataFrame(news_df)
        
        # 計算平均情緒分數
        avg_sentiment = df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0
        
        # 計算正面、負面、中性新聞比例
        if 'sentiment_score' in df.columns:
            positive_news = df[df['sentiment_score'] > 0.2]
            negative_news = df[df['sentiment_score'] < -0.2]
            neutral_news = df[(df['sentiment_score'] >= -0.2) & (df['sentiment_score'] <= 0.2)]
            
            total_news = len(df)
            
            return {
                'stock_id': stock_id,
                'news_count': total_news,
                'avg_sentiment': avg_sentiment,
                'positive_ratio': len(positive_news) / total_news if total_news > 0 else 0,
                'negative_ratio': len(negative_news) / total_news if total_news > 0 else 0,
                'neutral_ratio': len(neutral_news) / total_news if total_news > 0 else 0
            }
        else:
            return {
                'stock_id': stock_id,
                'news_count': len(df),
                'avg_sentiment': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0
            }
    
    def extract_hot_topics(self, days: int = 7, top_n: int = 5) -> List[Dict]:
        """提取熱門話題
        
        Args:
            days (int): 最近幾天
            top_n (int): 返回前N個話題
        
        Returns:
            list: 熱門話題列表
        """
        # 查詢最近的新聞標題
        query = """
        SELECT title
        FROM stock_news
        WHERE publish_date >= date('now', '-{} days')
        """.format(days)
        
        df = pd.read_sql(query, self.conn)
        
        if df.empty:
            return []
        
        # 使用TF-IDF提取關鍵詞
        tfidf = TfidfVectorizer(
            max_features=100, 
            stop_words=['的', '是', '在', '了', '和', '與', '及', '或', '但', '而', '所', '以', '等']
        )
        
        tfidf_matrix = tfidf.fit_transform(df['title'])
        feature_names = tfidf.get_feature_names_out()
        
        # 計算詞頻總和
        word_scores = tfidf_matrix.sum(axis=0).A1
        
        # 獲取分數最高的詞
        top_indices = word_scores.argsort()[-top_n:][::-1]
        top_words = [(feature_names[i], word_scores[i]) for i in top_indices]
        
        return [{'word': word, 'score': float(score)} for word, score in top_words]
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()