# news_api.py
from flask import Blueprint, request, jsonify
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
import json

# 創建 Blueprint
news_api = Blueprint('news_api', __name__)

# 設置日誌
logger = logging.getLogger("NewsAPI")

# 資料庫連接
def get_db_connection(database_path="tw_stock_data.db"):
    """建立資料庫連接"""
    return sqlite3.connect(database_path)

@news_api.route('/api/news/latest', methods=['GET'])
def get_latest_news():
    """獲取最新新聞列表
    
    參數:
        - days: 取得最近多少天的新聞 (預設: 7)
        - limit: 返回記錄數量上限 (預設: 20)
        - stock_id: (可選) 特定股票代碼
    """
    try:
        days = int(request.args.get('days', 7))
        limit = int(request.args.get('limit', 20))
        stock_id = request.args.get('stock_id')
        
        conn = get_db_connection()
        
        if stock_id:
            # 查詢特定股票相關新聞
            query = """
            SELECT n.id, n.title, n.content, n.url, n.publish_date, n.source,
                   s.sentiment_score
            FROM stock_news n
            JOIN news_stock_relation r ON n.id = r.news_id
            LEFT JOIN news_sentiment s ON n.id = s.news_id
            WHERE r.stock_id = ?
            AND n.publish_date >= date('now', '-? days')
            ORDER BY n.publish_date DESC, r.relevance DESC
            LIMIT ?
            """
            params = (stock_id, days, limit)
        else:
            # 查詢所有最新新聞
            query = """
            SELECT n.id, n.title, n.content, n.url, n.publish_date, n.source,
                   s.sentiment_score
            FROM stock_news n
            LEFT JOIN news_sentiment s ON n.id = s.news_id
            WHERE n.publish_date >= date('now', '-? days')
            ORDER BY n.publish_date DESC
            LIMIT ?
            """
            params = (days, limit)
        
        df = pd.read_sql(query, conn, params=params)
        
        # 關閉連接
        conn.close()
        
        # 處理結果
        if df.empty:
            return jsonify({
                "status": "success", 
                "message": "未找到相關新聞",
                "news": []
            })
        
        # 轉換日期格式
        df['publish_date'] = pd.to_datetime(df['publish_date']).dt.strftime('%Y-%m-%d')
        
        # 縮短內容長度（預覽用）
        df['content'] = df['content'].apply(lambda x: (x[:150] + '...') if pd.notna(x) and len(x) > 150 else x)
        
        # 添加情緒標籤
        def get_sentiment_label(score):
            if pd.isna(score):
                return "未分析"
            elif score > 0.2:
                return "正面"
            elif score < -0.2:
                return "負面"
            else:
                return "中性"
        
        df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
        
        # 轉換為字典列表
        news_list = df.to_dict('records')
        
        return jsonify({
            "status": "success",
            "count": len(news_list),
            "news": news_list
        })
    
    except Exception as e:
        logger.error(f"獲取最新新聞時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_api.route('/api/news/sentiment/summary', methods=['GET'])
def get_sentiment_summary():
    """獲取新聞情緒摘要
    
    參數:
        - stock_id: 股票代碼
        - days: 分析最近多少天的新聞 (預設: 30)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 30))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        conn = get_db_connection()
        
        # 查詢相關新聞
        query = """
        SELECT n.id, s.sentiment_score
        FROM stock_news n
        JOIN news_stock_relation r ON n.id = r.news_id
        LEFT JOIN news_sentiment s ON n.id = s.news_id
        WHERE r.stock_id = ?
        AND n.publish_date >= date('now', '-? days')
        """
        
        df = pd.read_sql(query, conn, params=(stock_id, days))
        
        # 關閉連接
        conn.close()
        
        # 處理結果
        if df.empty:
            return jsonify({
                "status": "success", 
                "message": "未找到相關新聞",
                "sentiment": {
                    "stock_id": stock_id,
                    "news_count": 0,
                    "avg_sentiment": 0,
                    "positive_ratio": 0,
                    "negative_ratio": 0,
                    "neutral_ratio": 0
                }
            })
        
        # 計算情緒摘要
        total_news = len(df)
        avg_sentiment = df['sentiment_score'].mean()
        
        # 計算正面、負面、中性新聞比例
        positive_news = df[df['sentiment_score'] > 0.2]
        negative_news = df[df['sentiment_score'] < -0.2]
        neutral_news = df[(df['sentiment_score'] >= -0.2) & (df['sentiment_score'] <= 0.2)]
        
        sentiment_summary = {
            'stock_id': stock_id,
            'news_count': total_news,
            'avg_sentiment': float(avg_sentiment) if not pd.isna(avg_sentiment) else 0,
            'positive_ratio': len(positive_news) / total_news if total_news > 0 else 0,
            'negative_ratio': len(negative_news) / total_news if total_news > 0 else 0,
            'neutral_ratio': len(neutral_news) / total_news if total_news > 0 else 0
        }
        
        return jsonify({
            "status": "success",
            "sentiment": sentiment_summary
        })
    
    except Exception as e:
        logger.error(f"獲取情緒摘要時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_api.route('/api/news/sentiment/trend', methods=['GET'])
def get_sentiment_trend():
    """獲取新聞情緒趨勢資料
    
    參數:
        - stock_id: 股票代碼
        - days: 分析最近多少天的新聞 (預設: 30)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 30))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        conn = get_db_connection()
        
        # 查詢相關新聞
        query = """
        SELECT n.publish_date, s.sentiment_score, r.relevance, COUNT(*) as news_count
        FROM stock_news n
        JOIN news_stock_relation r ON n.id = r.news_id
        LEFT JOIN news_sentiment s ON n.id = s.news_id
        WHERE r.stock_id = ?
        AND n.publish_date >= date('now', '-? days')
        GROUP BY n.publish_date
        ORDER BY n.publish_date
        """
        
        df = pd.read_sql(query, conn, params=(stock_id, days))
        
        # 關閉連接
        conn.close()
        
        # 處理結果
        if df.empty:
            return jsonify({
                "status": "success", 
                "message": "未找到相關新聞",
                "trend": []
            })
        
        # 計算加權情緒分數
        df['weighted_score'] = df['sentiment_score'] * df['relevance']
        
        # 轉換日期格式
        df['publish_date'] = pd.to_datetime(df['publish_date']).dt.strftime('%Y-%m-%d')
        
        # 轉換為字典列表
        trend_data = df[['publish_date', 'weighted_score', 'news_count']].to_dict('records')
        
        return jsonify({
            "status": "success",
            "trend": trend_data
        })
    
    except Exception as e:
        logger.error(f"獲取情緒趨勢時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_api.route('/api/news/keywords', methods=['GET'])
def get_news_keywords():
    """獲取新聞關鍵字統計
    
    參數:
        - stock_id: (可選) 股票代碼
        - days: 分析最近多少天的新聞 (預設: 30)
        - limit: 返回關鍵字數量 (預設: 20)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 30))
        limit = int(request.args.get('limit', 20))
        
        conn = get_db_connection()
        
        # 建立查詢
        if stock_id:
            query = """
            SELECT s.keywords
            FROM stock_news n
            JOIN news_sentiment s ON n.id = s.news_id
            JOIN news_stock_relation r ON n.id = r.news_id
            WHERE r.stock_id = ?
            AND n.publish_date >= date('now', '-? days')
            """
            params = (stock_id, days)
        else:
            query = """
            SELECT s.keywords
            FROM stock_news n
            JOIN news_sentiment s ON n.id = s.news_id
            WHERE n.publish_date >= date('now', '-? days')
            """
            params = (days,)
        
        # 執行查詢
        df = pd.read_sql(query, conn, params=params)
        
        # 關閉連接
        conn.close()
        
        # 處理結果
        if df.empty:
            return jsonify({
                "status": "success", 
                "message": "未找到關鍵字資料",
                "keywords": []
            })
        
        # 合併所有關鍵字並統計
        all_keywords = []
        for keywords_str in df['keywords']:
            if pd.notna(keywords_str) and keywords_str:
                all_keywords.extend(keywords_str.split(','))
        
        # 統計關鍵字頻率
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        
        # 轉換為列表
        keyword_list = [{"keyword": k, "count": c} for k, c in keyword_counts.most_common(limit)]
        
        return jsonify({
            "status": "success",
            "keywords": keyword_list
        })
    
    except Exception as e:
        logger.error(f"獲取關鍵字統計時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_api.route('/api/news/topics', methods=['GET'])
def get_news_topics():
    """獲取熱門新聞主題
    
    參數:
        - days: 分析最近多少天的新聞 (預設: 7)
        - limit: 返回主題數量 (預設: 5)
    """
    try:
        days = int(request.args.get('days', 7))
        limit = int(request.args.get('limit', 5))
        
        conn = get_db_connection()
        
        # 查詢最近的新聞
        query = """
        SELECT n.id, n.title, n.content, n.publish_date, n.source, s.keywords
        FROM stock_news n
        JOIN news_sentiment s ON n.id = s.news_id
        WHERE n.publish_date >= date('now', '-? days')
        """
        
        df = pd.read_sql(query, conn, params=(days,))
        
        # 關閉連接
        conn.close()
        
        # 處理結果
        if df.empty:
            return jsonify({
                "status": "success", 
                "message": "未找到主題資料",
                "topics": []
            })
        
        # 合併所有關鍵字並統計
        all_keywords = []
        for keywords_str in df['keywords']:
            if pd.notna(keywords_str) and keywords_str:
                all_keywords.extend(keywords_str.split(','))
        
        # 統計關鍵字頻率
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        
        # 獲取前N個關鍵詞
        top_keywords = keyword_counts.most_common(limit)
        
        # 為每個關鍵詞找出相關的新聞
        topics = []
        for keyword, count in top_keywords:
            related_news = []
            
            # 先合併關鍵字和內容
            df['combined_text'] = df['title'] + " " + df['content'].fillna("")
            
            # 過濾包含該關鍵字的新聞
            for _, row in df[df['combined_text'].str.contains(keyword, na=False, case=False)].head(3).iterrows():
                related_news.append({
                    'id': int(row['id']),
                    'title': row['title'],
                    'date': row['publish_date'],
                    'source': row['source']
                })
            
            topics.append({
                'keyword': keyword,
                'count': count,
                'related_news': related_news
            })
        
        return jsonify({
            "status": "success",
            "topics": topics
        })
    
    except Exception as e:
        logger.error(f"獲取熱門主題時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_api.route('/api/news/download', methods=['POST'])
def download_news():
    """下載和分析新聞
    
    參數:
        - days: 下載最近多少天的新聞 (預設: 7)
    """
    try:
        data = request.json
        days = int(data.get('days', 7))
        
        # 導入新聞收集器
        from data_collectors.news_collector import NewsCollector
        from analyzers.news_analyzer import NewsAnalyzer
        
        # 創建實例
        news_collector = NewsCollector()
        news_analyzer = NewsAnalyzer()
        
        # 下載新聞
        news_count = news_collector.collect_integrated_news(days)
        
        # 分析情緒
        sentiment_count = news_analyzer.analyze_news_sentiment()
        
        # 分析相關股票
        relation_count = news_analyzer.analyze_related_stocks()
        
        # 關閉資源
        news_collector.close()
        news_analyzer.close()
        
        return jsonify({
            "status": "success",
            "news_count": news_count,
            "sentiment_count": sentiment_count,
            "relation_count": relation_count,
            "message": f"成功下載 {news_count} 則新聞，分析了 {sentiment_count} 則情緒和 {relation_count} 則股票關係"
        })
    
    except Exception as e:
        logger.error(f"下載新聞時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_api.route('/api/news/detail/<int:news_id>', methods=['GET'])
def get_news_detail(news_id):
    """獲取單篇新聞詳情
    
    參數:
        - news_id: 新聞ID
    """
    try:
        conn = get_db_connection()
        
        # 查詢新聞詳情
        query = """
        SELECT n.id, n.title, n.content, n.url, n.publish_date, n.source,
               s.sentiment_score, s.positive_score, s.negative_score, s.neutral_score, s.keywords, s.summary
        FROM stock_news n
        LEFT JOIN news_sentiment s ON n.id = s.news_id
        WHERE n.id = ?
        """
        
        news_df = pd.read_sql(query, conn, params=(news_id,))
        
        # 查詢相關股票
        query_stocks = """
        SELECT r.stock_id, r.relevance, c.公司簡稱 as company_name
        FROM news_stock_relation r
        LEFT JOIN company_info c ON r.stock_id = c.stock_id
        WHERE r.news_id = ?
        ORDER BY r.relevance DESC
        """
        
        stocks_df = pd.read_sql(query_stocks, conn, params=(news_id,))
        
        # 關閉連接
        conn.close()
        
        # 處理結果
        if news_df.empty:
            return jsonify({
                "status": "error", 
                "message": "找不到指定的新聞"
            }), 404
        
        # 處理新聞資料
        news_data = news_df.iloc[0].to_dict()
        
        # 轉換日期格式
        news_data['publish_date'] = pd.to_datetime(news_data['publish_date']).strftime('%Y-%m-%d')
        
        # 處理關鍵字
        if pd.notna(news_data['keywords']) and news_data['keywords']:
            news_data['keywords'] = news_data['keywords'].split(',')
        else:
            news_data['keywords'] = []
        
        # 處理相關股票
        related_stocks = []
        for _, row in stocks_df.iterrows():
            stock_data = {
                'stock_id': row['stock_id'],
                'relevance': float(row['relevance']),
                'company_name': row['company_name'] if pd.notna(row['company_name']) else ""
            }
            related_stocks.append(stock_data)
        
        # 整合結果
        result = {
            "news": news_data,
            "related_stocks": related_stocks
        }
        
        return jsonify({
            "status": "success",
            "data": result
        })
    
    except Exception as e:
        logger.error(f"獲取新聞詳情時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500