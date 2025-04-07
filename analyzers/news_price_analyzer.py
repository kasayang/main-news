# news_price_analyzer.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats
import json
from typing import Dict, List, Optional, Tuple

class NewsPriceAnalyzer:
    """新聞與股價整合分析模組
    
    此模組用於分析新聞情緒和股價之間的關係，包括：
    - 新聞情緒對股價的延遲效應
    - 股價波動與新聞情緒的相關性
    - 重大新聞前後的股價行為
    """
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """初始化新聞與股價分析器
        
        Args:
            database_path (str): 資料庫路徑
        """
        self.database_path = database_path
        self.conn = sqlite3.connect(database_path)
        
        # 設置日誌
        self.logger = logging.getLogger("NewsPriceAnalyzer")
    
    def get_stock_price_data(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """獲取股票價格資料
        
        Args:
            stock_id (str): 股票代碼
            start_date (str): 開始日期
            end_date (str): 結束日期
        
        Returns:
            DataFrame: 股價資料
        """
        query = """
        SELECT date, open, high, low, close, volume
        FROM price_close
        WHERE stock_id = ?
        AND date BETWEEN ? AND ?
        ORDER BY date
        """
        
        return pd.read_sql(query, self.conn, params=(stock_id, start_date, end_date))
    
    def get_news_sentiment_data(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """獲取新聞情緒資料
        
        Args:
            stock_id (str): 股票代碼
            start_date (str): 開始日期
            end_date (str): 結束日期
        
        Returns:
            DataFrame: 新聞情緒資料
        """
        query = """
        SELECT n.publish_date as date, 
               AVG(s.sentiment_score * r.relevance) as sentiment_score,
               COUNT(*) as news_count
        FROM stock_news n
        JOIN news_stock_relation r ON n.id = r.news_id
        JOIN news_sentiment s ON n.id = s.news_id
        WHERE r.stock_id = ?
        AND n.publish_date BETWEEN ? AND ?
        GROUP BY n.publish_date
        ORDER BY n.publish_date
        """
        
        return pd.read_sql(query, self.conn, params=(stock_id, start_date, end_date))
    
    def analyze_correlation(self, stock_id: str, days: int = 90, lag_days: int = 5) -> Dict:
        """分析股價與新聞情緒的相關性
        
        Args:
            stock_id (str): 股票代碼
            days (int): 分析天數
            lag_days (int): 滯後天數範圍 (分析前後n天影響)
        
        Returns:
            Dict: 相關性分析結果
        """
        try:
            # 計算日期範圍
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days+lag_days)).strftime('%Y-%m-%d')
            
            # 獲取資料
            price_df = self.get_stock_price_data(stock_id, start_date, end_date)
            news_df = self.get_news_sentiment_data(stock_id, start_date, end_date)
            
            # 確保資料非空
            if price_df.empty or news_df.empty:
                self.logger.warning(f"股票 {stock_id} 缺少價格或新聞資料")
                return {"status": "error", "message": "資料不足"}
            
            # 轉換日期列
            price_df['date'] = pd.to_datetime(price_df['date'])
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            # 設置日期為索引
            price_df.set_index('date', inplace=True)
            news_df.set_index('date', inplace=True)
            
            # 填補缺失的日期 (無新聞的日期)
            # 創建所有日期的索引
            all_dates = pd.date_range(start=price_df.index.min(), end=price_df.index.max())
            
            # 重建新聞資料框架
            news_df = news_df.reindex(all_dates)
            
            # 填補缺失值
            news_df['news_count'].fillna(0, inplace=True)
            news_df['sentiment_score'].fillna(0, inplace=True)  # 無新聞的日子情緒設為中性
            
            # 計算每日價格變動百分比
            price_df['price_change'] = price_df['close'].pct_change() * 100
            
            # 分析不同滯後天數的相關性
            correlation_results = []
            p_value_results = []
            
            for lag in range(-lag_days, lag_days + 1):
                # 移動新聞情緒資料
                if lag < 0:
                    # 新聞滯後於股價 (新聞可能受股價影響)
                    shifted_sentiment = news_df['sentiment_score'].shift(-lag)
                    label = f"股價領先新聞{abs(lag)}天"
                else:
                    # 新聞領先股價 (股價可能受新聞影響)
                    shifted_sentiment = news_df['sentiment_score'].shift(lag)
                    label = f"新聞領先股價{lag}天" if lag > 0 else "同一天"
                
                # 計算相關係數和p值
                sentiment_corr = price_df['price_change'].corr(shifted_sentiment)
                # 使用scipy計算p值
                corr_p_value = stats.pearsonr(
                    price_df['price_change'].dropna(),
                    shifted_sentiment.dropna()[:len(price_df['price_change'].dropna())]
                )[1]
                
                correlation_results.append({
                    'lag': lag,
                    'label': label,
                    'correlation': sentiment_corr,
                    'p_value': corr_p_value
                })
                
                # 保存p值
                p_value_results.append(corr_p_value)
            
            # 找出最顯著的相關性
            best_correlation = max(correlation_results, key=lambda x: abs(x['correlation']))
            
            # 計算平均每日新聞數
            avg_daily_news = news_df['news_count'].mean()
            
            # 計算新聞量與股價波動的相關性
            volume_volatility_corr = news_df['news_count'].corr(price_df['close'].pct_change().abs())
            
            # 計算最近一個月的新聞情緒與股價變動趨勢
            recent_days = min(30, len(price_df))
            recent_price_trend = price_df['close'].pct_change().tail(recent_days).mean() * 100
            recent_sentiment_trend = news_df['sentiment_score'].tail(recent_days).mean()
            
            # 整理結果
            result = {
                "status": "success",
                "stock_id": stock_id,
                "analysis_period": {
                    "start_date": price_df.index.min().strftime('%Y-%m-%d'),
                    "end_date": price_df.index.max().strftime('%Y-%m-%d'),
                    "days": len(price_df)
                },
                "correlation_analysis": {
                    "lag_correlations": correlation_results,
                    "best_correlation": best_correlation,
                    "is_significant": best_correlation['p_value'] < 0.05
                },
                "news_statistics": {
                    "total_news": int(news_df['news_count'].sum()),
                    "avg_daily_news": float(avg_daily_news),
                    "news_volume_price_volatility_correlation": float(volume_volatility_corr)
                },
                "recent_trends": {
                    "price_change_percent": float(recent_price_trend),
                    "sentiment_score": float(recent_sentiment_trend)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析股價與情緒相關性時發生錯誤: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def identify_significant_news(self, stock_id: str, days: int = 90, price_change_threshold: float = 2.0) -> Dict:
        """識別重大新聞事件及其對股價的影響
        
        Args:
            stock_id (str): 股票代碼
            days (int): 分析天數
            price_change_threshold (float): 價格變動閾值 (百分比)
        
        Returns:
            Dict: 重大新聞事件分析
        """
        try:
            # 計算日期範圍
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 獲取股價資料
            price_df = self.get_stock_price_data(stock_id, start_date, end_date)
            
            # 確保資料非空
            if price_df.empty:
                self.logger.warning(f"股票 {stock_id} 缺少價格資料")
                return {"status": "error", "message": "股價資料不足"}
            
            # 計算每日股價變動百分比
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df['price_change'] = price_df['close'].pct_change() * 100
            
            # 識別價格顯著變動的日期
            significant_dates = price_df[abs(price_df['price_change']) > price_change_threshold]
            
            if significant_dates.empty:
                return {
                    "status": "success",
                    "stock_id": stock_id,
                    "message": "在指定時間段內未找到顯著的價格變動",
                    "significant_events": []
                }
            
            # 獲取這些日期前後的新聞
            significant_events = []
            
            for _, row in significant_dates.iterrows():
                event_date = row['date']
                price_change = row['price_change']
                
                # 提前3天到事後1天的日期範圍
                news_start_date = (event_date - timedelta(days=3)).strftime('%Y-%m-%d')
                news_end_date = (event_date + timedelta(days=1)).strftime('%Y-%m-%d')
                
                # 查詢這段時間的新聞
                query = """
                SELECT n.id, n.title, n.publish_date, s.sentiment_score, r.relevance
                FROM stock_news n
                JOIN news_stock_relation r ON n.id = r.news_id
                JOIN news_sentiment s ON n.id = s.news_id
                WHERE r.stock_id = ?
                AND n.publish_date BETWEEN ? AND ?
                ORDER BY n.publish_date, r.relevance DESC
                """
                
                news_df = pd.read_sql(query, self.conn, params=(stock_id, news_start_date, news_end_date))
                
                # 計算加權情緒
                if not news_df.empty:
                    news_df['weighted_score'] = news_df['sentiment_score'] * news_df['relevance']
                    avg_sentiment = news_df['weighted_score'].mean()
                    
                    # 獲取最相關的幾則新聞
                    top_news = news_df.sort_values('relevance', ascending=False).head(3)
                    
                    # 整理新聞列表
                    news_list = []
                    for _, news_row in top_news.iterrows():
                        news_list.append({
                            "id": int(news_row['id']),
                            "title": news_row['title'],
                            "publish_date": pd.to_datetime(news_row['publish_date']).strftime('%Y-%m-%d'),
                            "sentiment_score": float(news_row['sentiment_score']),
                            "relevance": float(news_row['relevance'])
                        })
                    
                    # 記錄此事件
                    significant_events.append({
                        "date": event_date.strftime('%Y-%m-%d'),
                        "price_change": float(price_change),
                        "news_count": len(news_df),
                        "avg_sentiment": float(avg_sentiment) if not pd.isna(avg_sentiment) else 0,
                        "related_news": news_list,
                        "is_aligned": (price_change > 0 and avg_sentiment > 0) or 
                                     (price_change < 0 and avg_sentiment < 0)
                    })
            
            # 計算新聞情緒與股價變動一致的比例
            aligned_count = sum(1 for event in significant_events if event['is_aligned'])
            alignment_ratio = aligned_count / len(significant_events) if significant_events else 0
            
            return {
                "status": "success",
                "stock_id": stock_id,
                "analysis_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "days": days
                },
                "summary": {
                    "significant_event_count": len(significant_events),
                    "news_price_alignment_ratio": alignment_ratio,
                    "price_change_threshold": price_change_threshold
                },
                "significant_events": significant_events
            }
            
        except Exception as e:
            self.logger.error(f"識別重大新聞事件時發生錯誤: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def predict_price_impact(self, stock_id: str, recent_days: int = 30, prediction_window: int = 5) -> Dict:
        """基於新聞情緒預測股價可能的影響
        
        Args:
            stock_id (str): 股票代碼
            recent_days (int): 參考最近天數的資料
            prediction_window (int): 預測的時間窗口 (天數)
        
        Returns:
            Dict: 預測結果
        """
        try:
            # 分析相關性
            corr_result = self.analyze_correlation(stock_id, days=90)
            
            if corr_result['status'] != 'success':
                return corr_result
            
            # 獲取最佳滯後天數
            best_corr = corr_result['correlation_analysis']['best_correlation']
            best_lag = best_corr['lag']
            is_significant = corr_result['correlation_analysis']['is_significant']
            
            # 如果相關性不顯著，返回警告
            if not is_significant:
                return {
                    "status": "warning",
                    "stock_id": stock_id,
                    "message": "新聞情緒與股價相關性不顯著，預測可能不準確",
                    "correlation": float(best_corr['correlation']),
                    "prediction": "中性",
                    "confidence": 0.0
                }
            
            # 獲取最近新聞情緒資料
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=recent_days)).strftime('%Y-%m-%d')
            
            news_df = self.get_news_sentiment_data(stock_id, start_date, end_date)
            
            if news_df.empty:
                return {
                    "status": "error",
                    "message": "缺少最近的新聞情緒資料"
                }
            
            # 計算最近的情緒趨勢
            recent_sentiment = news_df['sentiment_score'].mean()
            
            # 計算預測信心度 (基於相關性強度和相關性顯著性)
            prediction_confidence = min(abs(best_corr['correlation']) * (1 - best_corr['p_value']), 1.0)
            
            # 判斷預測方向
            prediction_direction = ""
            if abs(recent_sentiment) < 0.1:
                prediction_direction = "中性"
                prediction_confidence *= 0.5  # 降低中性預測的信心
            elif recent_sentiment > 0:
                # 如果最佳滯後為正，表示新聞領先股價，正面情緒預示股價上漲
                # 如果最佳滯後為負，表示股價領先新聞，正面情緒可能是股價已上漲的結果
                prediction_direction = "上漲" if best_lag >= 0 else "持平或短期修正"
            else:
                prediction_direction = "下跌" if best_lag >= 0 else "持平或短期反彈"
            
            # 準備預測結果
            result = {
                "status": "success",
                "stock_id": stock_id,
                "analysis": {
                    "correlation": float(best_corr['correlation']),
                    "best_lag": best_lag,
                    "recent_sentiment": float(recent_sentiment),
                    "news_count_last_days": int(news_df['news_count'].sum())
                },
                "prediction": {
                    "direction": prediction_direction,
                    "confidence": float(prediction_confidence),
                    "window_days": prediction_window,
                    "reasoning": f"基於{best_corr['label']}的相關性分析，近期的{'正面' if recent_sentiment > 0 else '負面' if recent_sentiment < 0 else '中性'}新聞情緒可能對股價產生{prediction_direction}影響"
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"預測股價影響時發生錯誤: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def analyze_sentiment_turning_points(self, stock_id: str, days: int = 90, smoothing_window: int = 5) -> Dict:
        """分析情緒轉折點與股價的關係
        
        Args:
            stock_id (str): 股票代碼
            days (int): 分析天數
            smoothing_window (int): 平滑窗口大小 (天數)
        
        Returns:
            Dict: 轉折點分析結果
        """
        try:
            # 計算日期範圍
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 獲取資料
            price_df = self.get_stock_price_data(stock_id, start_date, end_date)
            news_df = self.get_news_sentiment_data(stock_id, start_date, end_date)
            
            # 確保資料非空
            if price_df.empty or news_df.empty:
                self.logger.warning(f"股票 {stock_id} 缺少價格或新聞資料")
                return {"status": "error", "message": "資料不足"}
            
            # 轉換日期列
            price_df['date'] = pd.to_datetime(price_df['date'])
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            # 合併資料
            price_df.set_index('date', inplace=True)
            news_df.set_index('date', inplace=True)
            
            # 對所有日期重建索引
            all_dates = pd.date_range(start=price_df.index.min(), end=price_df.index.max())
            price_df = price_df.reindex(all_dates)
            news_df = news_df.reindex(all_dates)
            
            # 向前填充股價
            price_df['close'] = price_df['close'].fillna(method='ffill')
            
            # 填充缺失的新聞情緒
            news_df['sentiment_score'] = news_df['sentiment_score'].fillna(0)
            news_df['news_count'] = news_df['news_count'].fillna(0)
            
            # 平滑情緒曲線
            news_df['smooth_sentiment'] = news_df['sentiment_score'].rolling(window=smoothing_window, center=True).mean().fillna(news_df['sentiment_score'])
            
            # 找出情緒轉折點 (從正轉負或從負轉正)
            turning_points = []
            
            for i in range(1, len(news_df) - 1):
                prev_sentiment = news_df['smooth_sentiment'].iloc[i-1]
                curr_sentiment = news_df['smooth_sentiment'].iloc[i]
                next_sentiment = news_df['smooth_sentiment'].iloc[i+1]
                
                # 如果前後情緒符號變化 (過零點)
                if (prev_sentiment * next_sentiment < 0) or \
                   (prev_sentiment * curr_sentiment < 0) or \
                   (curr_sentiment * next_sentiment < 0):
                    turning_date = news_df.index[i]
                    turning_points.append({
                        "date": turning_date,
                        "from_sentiment": float(prev_sentiment),
                        "to_sentiment": float(next_sentiment),
                        "direction": "正轉負" if prev_sentiment > 0 and next_sentiment < 0 else "負轉正"
                    })
            
            # 分析轉折點前後的股價變化
            turning_point_analysis = []
            
            for point in turning_points:
                turning_date = point["date"]
                
                # 獲取轉折點前後5/10天的股價
                before_5d = turning_date - timedelta(days=5)
                before_5d_price = price_df.loc[before_5d:turning_date].iloc[0]['close'] if before_5d in price_df.index else None
                
                after_5d = turning_date + timedelta(days=5)
                after_5d_price = price_df.loc[turning_date:after_5d].iloc[-1]['close'] if after_5d in price_df.index else None
                
                after_10d = turning_date + timedelta(days=10)
                after_10d_price = price_df.loc[turning_date:after_10d].iloc[-1]['close'] if after_10d in price_df.index else None
                
                # 計算變化百分比
                price_change_5d = ((after_5d_price / price_df.loc[turning_date]['close']) - 1) * 100 if after_5d_price and turning_date in price_df.index else None
                price_change_10d = ((after_10d_price / price_df.loc[turning_date]['close']) - 1) * 100 if after_10d_price and turning_date in price_df.index else None
                
                # 判斷情緒與股價變化是否一致
                is_aligned_5d = False
                is_aligned_10d = False
                
                if price_change_5d is not None:
                    is_aligned_5d = (point["direction"] == "負轉正" and price_change_5d > 0) or \
                                   (point["direction"] == "正轉負" and price_change_5d < 0)
                
                if price_change_10d is not None:
                    is_aligned_10d = (point["direction"] == "負轉正" and price_change_10d > 0) or \
                                    (point["direction"] == "正轉負" and price_change_10d < 0)
                
                # 獲取轉折點前後的新聞
                news_start = (turning_date - timedelta(days=3)).strftime('%Y-%m-%d')
                news_end = (turning_date + timedelta(days=1)).strftime('%Y-%m-%d')
                
                query = """
                SELECT n.id, n.title, n.publish_date
                FROM stock_news n
                JOIN news_stock_relation r ON n.id = r.news_id
                WHERE r.stock_id = ?
                AND n.publish_date BETWEEN ? AND ?
                ORDER BY r.relevance DESC
                LIMIT 3
                """
                
                related_news_df = pd.read_sql(query, self.conn, params=(stock_id, news_start, news_end))
                
                related_news = []
                for _, row in related_news_df.iterrows():
                    related_news.append({
                        "id": int(row['id']),
                        "title": row['title'],
                        "publish_date": pd.to_datetime(row['publish_date']).strftime('%Y-%m-%d')
                    })
                
                # 添加到分析結果
                turning_point_analysis.append({
                    "date": turning_date.strftime('%Y-%m-%d'),
                    "direction": point["direction"],
                    "price_change_5d": float(price_change_5d) if price_change_5d is not None else None,
                    "price_change_10d": float(price_change_10d) if price_change_10d is not None else None,
                    "is_aligned_5d": bool(is_aligned_5d),
                    "is_aligned_10d": bool(is_aligned_10d),
                    "related_news": related_news
                })
            
            # 計算一致性比例
            aligned_5d_count = sum(1 for point in turning_point_analysis if point['is_aligned_5d'])
            aligned_10d_count = sum(1 for point in turning_point_analysis if point['is_aligned_10d'])
            
            alignment_ratio_5d = aligned_5d_count / len(turning_point_analysis) if turning_point_analysis else 0
            alignment_ratio_10d = aligned_10d_count / len(turning_point_analysis) if turning_point_analysis else 0
            
            return {
                "status": "success",
                "stock_id": stock_id,
                "analysis_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "days": days
                },
                "summary": {
                    "turning_point_count": len(turning_point_analysis),
                    "alignment_ratio_5d": float(alignment_ratio_5d),
                    "alignment_ratio_10d": float(alignment_ratio_10d)
                },
                "turning_points": turning_point_analysis
            }
            
        except Exception as e:
            self.logger.error(f"分析情緒轉折點時發生錯誤: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()