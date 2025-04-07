# news_summarizer_api.py
from flask import Blueprint, request, jsonify
import logging
from news_summarizer import NewsSummarizer

# 創建 Blueprint
news_summarizer_api = Blueprint('news_summarizer_api', __name__)

# 設置日誌
logger = logging.getLogger("NewsSummarizerAPI")

@news_summarizer_api.route('/api/news/summary/stock', methods=['GET'])
def get_stock_news_summary():
    """獲取特定股票的新聞摘要
    
    參數:
        - stock_id: 股票代碼 (必填)
        - days: 分析最近幾天的新聞 (預設: 7)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 7))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        summarizer = NewsSummarizer()
        result = summarizer.generate_stock_news_report(stock_id, days)
        summarizer.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"生成股票新聞摘要時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_summarizer_api.route('/api/news/summary/timeline', methods=['GET'])
def get_timeline_summary():
    """獲取股票新聞的時間線摘要
    
    參數:
        - stock_id: 股票代碼 (必填)
        - days: 分析最近幾天的新聞 (預設: 30)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 30))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        summarizer = NewsSummarizer()
        result = summarizer.generate_timeline_summary(stock_id, days)
        summarizer.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"生成時間線摘要時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_summarizer_api.route('/api/news/summary/topic', methods=['GET'])
def get_topic_summary():
    """獲取股票新聞的主題摘要
    
    參數:
        - stock_id: 股票代碼 (必填)
        - days: 分析最近幾天的新聞 (預設: 7)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 7))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        summarizer = NewsSummarizer()
        result = summarizer.generate_topic_summary(stock_id, days)
        summarizer.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"生成主題摘要時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_summarizer_api.route('/api/news/summary/market', methods=['GET'])
def get_market_summary():
    """獲取市場整體新聞摘要
    
    參數:
        - days: 分析最近幾天的新聞 (預設: 3)
    """
    try:
        days = int(request.args.get('days', 3))
        
        summarizer = NewsSummarizer()
        result = summarizer.generate_market_news_report(days)
        summarizer.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"生成市場摘要時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_summarizer_api.route('/api/news/summary/keywords', methods=['POST'])
def get_keywords_summary():
    """根據關鍵字獲取相關新聞摘要
    
    JSON 參數:
        - keywords: 關鍵字列表
        - days: 分析最近幾天的新聞 (預設: 7)
    """
    try:
        data = request.json
        
        if not data or 'keywords' not in data:
            return jsonify({
                "status": "error",
                "message": "必須提供關鍵字列表"
            }), 400
        
        keywords = data.get('keywords', [])
        days = int(data.get('days', 7))
        
        if not keywords or not isinstance(keywords, list):
            return jsonify({
                "status": "error",
                "message": "關鍵字必須是非空列表"
            }), 400
        
        summarizer = NewsSummarizer()
        
        # 獲取相關新聞
        news_df = summarizer.get_news_by_topic(keywords, days)
        
        if news_df.empty:
            result = {
                "status": "error",
                "message": "未找到與關鍵字相關的新聞"
            }
        else:
            # 生成摘要
            summary = summarizer.generate_extractive_summary(news_df)
            
            # 發現主題
            topics = summarizer.discover_topics(news_df)
            
            result = {
                "status": "success",
                "keywords": keywords,
                "news_count": len(news_df),
                "date_range": summary['date_range'],
                "summary": summary['summary'],
                "key_phrases": summary['key_phrases'],
                "topics": topics
            }
        
        summarizer.close()
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"生成關鍵字摘要時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500