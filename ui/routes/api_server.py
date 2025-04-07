# api_server.py
from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import datetime
import logging
import traceback
from werkzeug.serving import run_simple

# 導入模型
from models.combined_model import CombinedModel
from models.news_model import NewsModel
from analyzers.news_analyzer import NewsAnalyzer

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StockApiServer")

# 初始化Flask應用
app = Flask(__name__)

# 全局變數
DB_PATH = "tw_stock_data.db"
MODEL_DIR = "models"
predictor = None
news_model = None
news_analyzer = None

# 載入模型
def load_models():
    global predictor, news_model, news_analyzer
    try:
        predictor = CombinedModel(database_path=DB_PATH, model_dir=MODEL_DIR)
        news_model = NewsModel(database_path=DB_PATH, model_dir=MODEL_DIR)
        news_analyzer = NewsAnalyzer(database_path=DB_PATH)
        logger.info("模型載入成功")
    except Exception as e:
        logger.error(f"模型載入失敗: {str(e)}")
        logger.error(traceback.format_exc())

# API根路徑
@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "股票預測API服務運行中",
        "version": "1.0.0"
    })

# 股票預測API
@app.route('/api/predict', methods=['POST'])
def predict_stock():
    try:
        # 取得請求資料
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "未提供資料"}), 400
        
        stock_id = data.get('stock_id')
        prediction_days = int(data.get('prediction_days', 1))
        model_type = data.get('model_type', 'combined')
        
        if not stock_id:
            return jsonify({"status": "error", "message": "未提供股票代碼"}), 400
        
        # 確保模型已載入
        if predictor is None:
            load_models()
        
        # 根據模型類型選擇不同的預測方法
        if model_type == "price_only":
            # 只使用價格模型
            result = predictor.price_predictor.predict_price_movement(stock_id, prediction_days=prediction_days)
        elif model_type == "news_only":
            # 只使用新聞模型
            direction, confidence = predictor.news_model.predict_price_movement_from_news(stock_id)
            result = {
                'stock_id': stock_id,
                'prediction_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'target_date': (datetime.datetime.now() + datetime.timedelta(days=prediction_days)).strftime('%Y-%m-%d'),
                'prediction_days': prediction_days,
                'prediction': 'up' if direction > 0 else 'down',
                'probability': confidence,
                'model_type': "news_only"
            }
        else:
            # 使用綜合模型
            result = predictor.predict_price_movement(stock_id, prediction_days=prediction_days)
        
        if result:
            # 取得技術與新聞特徵
            tech_features = get_tech_features(stock_id)
            news_features = predictor.news_model.get_sentiment_features(stock_id)
            
            # 添加到結果中
            result['tech_features'] = tech_features
            result['news_features'] = news_features
            
            return jsonify({
                "status": "success",
                "result": result
            })
        else:
            return jsonify({
                "status": "error",
                "message": "預測失敗，請檢查股票代碼或資料可用性"
            }), 404
    
    except Exception as e:
        logger.error(f"預測時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

# 批次預測API
@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # 取得請求資料
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "未提供資料"}), 400
        
        stock_ids = data.get('stock_ids', [])
        prediction_days = int(data.get('prediction_days', 1))
        
        if not stock_ids:
            return jsonify({"status": "error", "message": "未提供股票代碼清單"}), 400
        
        # 確保模型已載入
        if predictor is None:
            load_models()
        
        # 執行批次預測
        results = []
        for stock_id in stock_ids:
            try:
                # 使用綜合模型預測
                prediction = predictor.predict_price_movement(stock_id, prediction_days=prediction_days)
                
                if prediction:
                    # 添加到結果列表
                    results.append(prediction)
            except Exception as stock_error:
                logger.error(f"預測 {stock_id} 時發生錯誤: {stock_error}")
                # 添加錯誤結果
                results.append({
                    'stock_id': stock_id,
                    'prediction_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                    'target_date': '-',
                    'prediction_days': prediction_days,
                    'prediction': 'error',
                    'probability': 0,
                    'error': str(stock_error)
                })
        
        return jsonify({
            "status": "success",
            "results": results
        })
    
    except Exception as e:
        logger.error(f"批次預測時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

# 股票新聞API
@app.route('/api/news/<stock_id>', methods=['GET'])
def get_stock_news(stock_id):
    try:
        days = int(request.args.get('days', 30))
        
        # 確保模型已載入
        if news_model is None:
            load_models()
        
        # 取得相關新聞
        news_list = news_model.get_related_news(stock_id, days)
        
        return jsonify({
            "status": "success",
            "stock_id": stock_id,
            "news": news_list
        })
    
    except Exception as e:
        logger.error(f"取得新聞時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

# 情緒摘要API
@app.route('/api/sentiment/<stock_id>', methods=['GET'])
def get_stock_sentiment(stock_id):
    try:
        days = int(request.args.get('days', 30))
        
        # 確保模型已載入
        if news_model is None:
            load_models()
        
        # 取得情緒摘要
        sentiment_summary = news_model.get_sentiment_summary(stock_id, days)
        
        return jsonify({
            "status": "success",
            "stock_id": stock_id,
            "sentiment": sentiment_summary
        })
    
    except Exception as e:
        logger.error(f"取得情緒摘要時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

# 下載新聞API
@app.route('/api/news/download', methods=['POST'])
def download_news():
    try:
        data = request.json
        days = int(data.get('days', 7))
        
        # 確保分析器已載入
        if news_analyzer is None:
            load_models()
        
        # 執行新聞下載和分析
        from data_collectors.news_collector import NewsCollector
        collector = NewsCollector(database_path=DB_PATH)
        
        # 下載新聞
        news_count = collector.collect_twse_news(days)
        
        # 分析情緒
        sentiment_count = news_analyzer.analyze_news_sentiment()
        
        # 分析相關股票
        relation_count = news_analyzer.analyze_related_stocks()
        
        # 關閉資源
        collector.close()
        
        return jsonify({
            "status": "success",
            "news_count": news_count,
            "sentiment_count": sentiment_count,
            "relation_count": relation_count,
            "message": f"成功下載 {news_count} 則新聞，分析了 {sentiment_count} 則情緒和 {relation_count} 則股票關係"
        })
    
    except Exception as e:
        logger.error(f"下載新聞時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

# 熱門主題API
@app.route('/api/news/topics', methods=['GET'])
def get_hot_topics():
    try:
        days = int(request.args.get('days', 7))
        
        # 確保分析器已載入
        if news_analyzer is None:
            load_models()
        
        # 取得熱門主題
        topics = news_analyzer.analyze_news_topics(days)
        
        return jsonify({
            "status": "success",
            "topics": topics
        })
    
    except Exception as e:
        logger.error(f"取得熱門主題時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

# 輔助函數：取得技術指標
def get_tech_features(stock_id):
    try:
        conn = predictor.conn
        
        # 查詢最近的技術指標數據
        query = f"""
        SELECT date, close, 
               (SELECT ma5 FROM price_close WHERE stock_id = '{stock_id}' ORDER BY date DESC LIMIT 1) as ma5,
               (SELECT ma10 FROM price_close WHERE stock_id = '{stock_id}' ORDER BY date DESC LIMIT 1) as ma10,
               (SELECT ma20 FROM price_close WHERE stock_id = '{stock_id}' ORDER BY date DESC LIMIT 1) as ma20,
               (SELECT RSI FROM technical_indicators WHERE stock_id = '{stock_id}' ORDER BY date DESC LIMIT 1) as rsi,
               (SELECT MACD FROM technical_indicators WHERE stock_id = '{stock_id}' ORDER BY date DESC LIMIT 1) as macd,
               (SELECT volume FROM price_volume WHERE stock_id = '{stock_id}' ORDER BY date DESC LIMIT 1) as volume,
               (SELECT AVG(volume) FROM price_volume WHERE stock_id = '{stock_id}' ORDER BY date DESC LIMIT 10) as avg_volume
        FROM price_close
        WHERE stock_id = '{stock_id}'
        ORDER BY date DESC
        LIMIT 1
        """
        
        df = pd.read_sql(query, conn)
        
        if df.empty:
            return []
        
        # 整理技術指標
        tech_features = []
        
        # 收盤價與均線關係
        close = df['close'].iloc[0]
        ma5 = df['ma5'].iloc[0]
        ma10 = df['ma10'].iloc[0]
        ma20 = df['ma20'].iloc[0]
        
        if pd.notna(ma5) and pd.notna(ma10):
            if ma5 > ma10:
                ma_signal = "偏多"
            else:
                ma_signal = "偏空"
            tech_features.append({
                "name": "5日/10日均線",
                "value": f"{ma5:.2f}/{ma10:.2f}",
                "signal": ma_signal
            })
        
        if pd.notna(close) and pd.notna(ma20):
            if close > ma20:
                price_signal = "偏多"
            else:
                price_signal = "偏空"
            tech_features.append({
                "name": "價格/20日均線",
                "value": f"{close:.2f}/{ma20:.2f}",
                "signal": price_signal
            })
        
        # RSI值
        rsi = df['rsi'].iloc[0]
        if pd.notna(rsi):
            if rsi > 70:
                rsi_signal = "超買"
            elif rsi < 30:
                rsi_signal = "超賣"
            else:
                rsi_signal = "中性"
            tech_features.append({
                "name": "RSI",
                "value": f"{rsi:.2f}",
                "signal": rsi_signal
            })
        
        # MACD
        macd = df['macd'].iloc[0]
        if pd.notna(macd):
            if macd > 0:
                macd_signal = "偏多"
            else:
                macd_signal = "偏空"
            tech_features.append({
                "name": "MACD",
                "value": f"{macd:.4f}",
                "signal": macd_signal
            })
        
        # 成交量變化
        volume = df['volume'].iloc[0]
        avg_volume = df['avg_volume'].iloc[0]
        if pd.notna(volume) and pd.notna(avg_volume) and avg_volume > 0:
            vol_ratio = volume / avg_volume
            if vol_ratio > 1.5:
                vol_signal = "放大"
            elif vol_ratio < 0.7:
                vol_signal = "縮小"
            else:
                vol_signal = "正常"
            tech_features.append({
                "name": "成交量比例",
                "value": f"{vol_ratio:.2f}",
                "signal": vol_signal
            })
        
        return tech_features
    
    except Exception as e:
        logger.error(f"獲取技術指標時發生錯誤: {str(e)}")
        return []

# 訓練模型API
@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        stock_id = data.get('stock_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        prediction_days = int(data.get('prediction_days', 1))
        
        if not stock_id:
            return jsonify({"status": "error", "message": "未提供股票代碼"}), 400
        
        # 確保模型已載入
        if predictor is None:
            load_models()
        
        # 執行訓練
        result = predictor.train_model(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            prediction_days=prediction_days
        )
        
        if result:
            # 整理訓練結果
            model_info = {
                'stock_id': stock_id,
                'prediction_days': prediction_days,
                'accuracy': float(result['accuracy']),
                'data_size': int(result['data_size']),
                'model_path': result['model_path']
            }
            
            # 如果有子模型準確率
            if 'tech_accuracy' in result:
                model_info['tech_accuracy'] = float(result['tech_accuracy'])
            
            if 'news_accuracy' in result:
                model_info['news_accuracy'] = float(result['news_accuracy'])
            
            return jsonify({
                "status": "success",
                "model_info": model_info
            })
        else:
            return jsonify({
                "status": "error",
                "message": "模型訓練失敗，請檢查資料是否足夠"
            }), 400
    
    except Exception as e:
        logger.error(f"訓練模型時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

# 應用程式啟動
if __name__ == '__main__':
    # 載入模型
    load_models()
    
    # 啟動服務
    port = 5001
    host = '0.0.0.0'
    logger.info(f"API服務啟動於 http://{host}:{port}")
    
    run_simple(host, port, app, use_reloader=True, use_debugger=True)