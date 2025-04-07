# news_price_api.py
from flask import Blueprint, request, jsonify
import logging
from news_price_analyzer import NewsPriceAnalyzer

# 創建 Blueprint
news_price_api = Blueprint('news_price_api', __name__)

# 設置日誌
logger = logging.getLogger("NewsPriceAPI")

@news_price_api.route('/api/news-price/correlation', methods=['GET'])
def get_correlation_analysis():
    """獲取新聞情緒與股價相關性分析
    
    參數:
        - stock_id: 股票代碼 (必填)
        - days: 分析天數 (預設: 90)
        - lag_days: 滯後天數範圍 (預設: 5)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 90))
        lag_days = int(request.args.get('lag_days', 5))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        analyzer = NewsPriceAnalyzer()
        result = analyzer.analyze_correlation(stock_id, days, lag_days)
        analyzer.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"處理相關性分析請求時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_price_api.route('/api/news-price/significant-events', methods=['GET'])
def get_significant_events():
    """獲取重大新聞事件分析
    
    參數:
        - stock_id: 股票代碼 (必填)
        - days: 分析天數 (預設: 90)
        - threshold: 股價變動閾值 (預設: 2.0)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 90))
        threshold = float(request.args.get('threshold', 2.0))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        analyzer = NewsPriceAnalyzer()
        result = analyzer.identify_significant_news(stock_id, days, threshold)
        analyzer.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"處理重大事件分析請求時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_price_api.route('/api/news-price/impact-prediction', methods=['GET'])
def get_impact_prediction():
    """基於新聞情緒預測股價可能的影響
    
    參數:
        - stock_id: 股票代碼 (必填)
        - recent_days: 參考最近天數 (預設: 30)
        - prediction_window: 預測窗口天數 (預設: 5)
    """
    try:
        stock_id = request.args.get('stock_id')
        recent_days = int(request.args.get('recent_days', 30))
        prediction_window = int(request.args.get('prediction_window', 5))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        analyzer = NewsPriceAnalyzer()
        result = analyzer.predict_price_impact(stock_id, recent_days, prediction_window)
        analyzer.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"處理影響預測請求時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@news_price_api.route('/api/news-price/turning-points', methods=['GET'])
def get_turning_points():
    """獲取情緒轉折點分析
    
    參數:
        - stock_id: 股票代碼 (必填)
        - days: 分析天數 (預設: 90)
        - smoothing_window: 平滑窗口大小 (預設: 5)
    """
    try:
        stock_id = request.args.get('stock_id')
        days = int(request.args.get('days', 90))
        smoothing_window = int(request.args.get('smoothing_window', 5))
        
        if not stock_id:
            return jsonify({
                "status": "error",
                "message": "必須提供股票代碼"
            }), 400
        
        analyzer = NewsPriceAnalyzer()
        result = analyzer.analyze_sentiment_turning_points(stock_id, days, smoothing_window)
        analyzer.close()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"處理轉折點分析請求時發生錯誤: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500