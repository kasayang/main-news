# web_app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 設定 matplotlib 在後台運行
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import requests
from datetime import datetime
import traceback

# 初始化 Flask 應用
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'stock_prediction_webapp_secret_key'

# API 服務器地址
API_BASE_URL = "http://localhost:5000"

# 首頁
@app.route('/')
def index():
    return render_template('index.html')

# 股票預測頁面
@app.route('/predict')
def predict():
    return render_template('predict.html')

# 執行預測
@app.route('/api/run_prediction', methods=['POST'])
def run_prediction():
    try:
        # 獲取表單數據
        stock_id = request.form.get('stock_id')
        prediction_days = int(request.form.get('prediction_days', 1))
        model_type = request.form.get('model_type', 'combined')
        
        if not stock_id:
            flash('請輸入股票代碼', 'danger')
            return redirect(url_for('predict'))
        
        # 呼叫 API 服務
        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json={
                "stock_id": stock_id,
                "prediction_days": prediction_days,
                "model_type": model_type
            }
        )
        
        # 檢查回應
        if response.status_code != 200:
            flash(f'API 請求失敗: {response.text}', 'danger')
            return redirect(url_for('predict'))
        
        result = response.json()
        
        if result.get('status') != 'success':
            flash(f'預測失敗: {result.get("message")}', 'danger')
            return redirect(url_for('predict'))
        
        # 生成預測圖表
        img_data = generate_prediction_chart(result)
        
        # 保存結果到 session
        prediction_data = result['result']
        
        return render_template(
            'prediction_result.html',
            stock_id=stock_id,
            prediction_days=prediction_days,
            result=prediction_data,
            chart_image=img_data
        )
    
    except Exception as e:
        flash(f'處理預測時發生錯誤: {str(e)}', 'danger')
        traceback.print_exc()
        return redirect(url_for('predict'))

# 批次預測頁面
@app.route('/batch_predict')
def batch_predict():
    return render_template('batch_predict.html')

# 執行批次預測
@app.route('/api/run_batch_prediction', methods=['POST'])
def run_batch_prediction():
    try:
        # 獲取表單數據
        stock_ids_text = request.form.get('stock_ids')
        prediction_days = int(request.form.get('prediction_days', 1))
        
        if not stock_ids_text:
            flash('請輸入股票代碼清單', 'danger')
            return redirect(url_for('batch_predict'))
        
        # 解析股票代碼
        stock_ids = [s.strip() for s in stock_ids_text.split(',')]
        
        # 呼叫 API 服務
        response = requests.post(
            f"{API_BASE_URL}/api/batch_predict",
            json={
                "stock_ids": stock_ids,
                "prediction_days": prediction_days
            }
        )
        
        # 檢查回應
        if response.status_code != 200:
            flash(f'API 請求失敗: {response.text}', 'danger')
            return redirect(url_for('batch_predict'))
        
        result = response.json()
        
        if result.get('status') != 'success':
            flash(f'批次預測失敗: {result.get("message")}', 'danger')
            return redirect(url_for('batch_predict'))
        
        # 生成批次預測圖表
        img_data = generate_batch_prediction_chart(result)
        
        # 將結果轉換為 DataFrame 以便於在網頁上顯示
        batch_results = result['results']
        
        return render_template(
            'batch_prediction_result.html',
            prediction_days=prediction_days,
            results=batch_results,
            chart_image=img_data
        )
    
    except Exception as e:
        flash(f'處理批次預測時發生錯誤: {str(e)}', 'danger')
        traceback.print_exc()
        return redirect(url_for('batch_predict'))

# 新聞分析頁面
@app.route('/news')
def news():
    return render_template('news.html')

# 獲取股票新聞
@app.route('/api/get_news', methods=['POST'])
def get_news():
    try:
        # 獲取表單數據
        stock_id = request.form.get('stock_id')
        days = int(request.form.get('days', 30))
        
        if not stock_id:
            flash('請輸入股票代碼', 'danger')
            return redirect(url_for('news'))
        
        # 呼叫 API 服務獲取新聞
        news_response = requests.get(f"{API_BASE_URL}/api/news/{stock_id}?days={days}")
        
        # 呼叫 API 服務獲取情緒摘要
        sentiment_response = requests.get(f"{API_BASE_URL}/api/sentiment/{stock_id}?days={days}")
        
        # 檢查回應
        if news_response.status_code != 200 or sentiment_response.status_code != 200:
            flash(f'API 請求失敗', 'danger')
            return redirect(url_for('news'))
        
        news_result = news_response.json()
        sentiment_result = sentiment_response.json()
        
        if news_result.get('status') != 'success' or sentiment_result.get('status') != 'success':
            flash(f'獲取新聞失敗', 'danger')
            return redirect(url_for('news'))
        
        # 生成新聞情緒圖表
        img_data = generate_news_sentiment_chart(stock_id, news_result, sentiment_result)
        
        # 獲取新聞列表
        news_list = news_result.get('news', [])
        sentiment_summary = sentiment_result.get('sentiment', {})
        
        return render_template(
            'news_result.html',
            stock_id=stock_id,
            days=days,
            news_list=news_list,
            sentiment_summary=sentiment_summary,
            chart_image=img_data
        )
    
    except Exception as e:
        flash(f'處理新聞請求時發生錯誤: {str(e)}', 'danger')
        traceback.print_exc()
        return redirect(url_for('news'))

# 下載新聞
@app.route('/api/download_news', methods=['POST'])
def download_news():
    try:
        # 獲取表單數據
        days = int(request.form.get('days', 7))
        
        # 呼叫 API 服務
        response = requests.post(
            f"{API_BASE_URL}/api/news/download",
            json={"days": days}
        )
        
        # 檢查回應
        if response.status_code != 200:
            return jsonify({"status": "error", "message": f"API 請求失敗: {response.text}"})
        
        result = response.json()
        
        if result.get('status') != 'success':
            return jsonify({"status": "error", "message": f"下載新聞失敗: {result.get('message')}"})
        
        return jsonify({"status": "success", "result": result})
    
    except Exception as e:
        return jsonify({"status": "error", "message": f"處理下載新聞請求時發生錯誤: {str(e)}"})

# 熱門主題
@app.route('/topics')
def topics():
    try:
        # 呼叫 API 服務
        response = requests.get(f"{API_BASE_URL}/api/news/topics")
        
        # 檢查回應
        if response.status_code != 200:
            flash(f'API 請求失敗: {response.text}', 'danger')
            return render_template('topics.html', topics=[])
        
        result = response.json()
        
        if result.get('status') != 'success':
            flash(f'獲取熱門主題失敗: {result.get("message")}', 'danger')
            return render_template('topics.html', topics=[])
        
        topics_list = result.get('topics', [])
        
        return render_template('topics.html', topics=topics_list)
    
    except Exception as e:
        flash(f'處理熱門主題請求時發生錯誤: {str(e)}', 'danger')
        traceback.print_exc()
        return render_template('topics.html', topics=[])

# 模型訓練頁面
@app.route('/train')
def train():
    return render_template('train.html')

# 執行模型訓練
@app.route('/api/run_training', methods=['POST'])
def run_training():
    try:
        # 獲取表單數據
        stock_id = request.form.get('stock_id')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        prediction_days = int(request.form.get('prediction_days', 1))
        
        if not stock_id:
            flash('請輸入股票代碼', 'danger')
            return redirect(url_for('train'))
        
        # 呼叫 API 服務
        response = requests.post(
            f"{API_BASE_URL}/api/train",
            json={
                "stock_id": stock_id,
                "start_date": start_date,
                "end_date": end_date,
                "prediction_days": prediction_days
            }
        )
        
        # 檢查回應
        if response.status_code != 200:
            flash(f'API 請求失敗: {response.text}', 'danger')
            return redirect(url_for('train'))
        
        result = response.json()
        
        if result.get('status') != 'success':
            flash(f'訓練模型失敗: {result.get("message")}', 'danger')
            return redirect(url_for('train'))
        
        # 獲取訓練結果
        model_info = result.get('model_info', {})
        
        return render_template(
            'training_result.html',
            stock_id=stock_id,
            prediction_days=prediction_days,
            model_info=model_info
        )
    
    except Exception as e:
        flash(f'處理訓練請求時發生錯誤: {str(e)}', 'danger')
        traceback.print_exc()
        return redirect(url_for('train'))

# ===== 生成圖表函數 =====

def generate_prediction_chart(result):
    """生成預測結果圖表"""
    try:
        if result.get('status') != 'success' or 'result' not in result:
            return None
        
        prediction_data = result['result']
        stock_id = prediction_data['stock_id']
        prediction = prediction_data['prediction']
        probability = prediction_data['probability']
        
        # 取得技術特徵和新聞特徵
        tech_features = prediction_data.get('tech_features', [])
        news_features = prediction_data.get('news_features', {})
        
        # 創建圖表
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # 技術指標視覺化
        ax1 = axes[0]
        if tech_features:
            feature_names = [f"{feature['name']}" for feature in tech_features]
            feature_values = [1 if "多" in feature['signal'] or "買" in feature['signal'] else 
                            -1 if "空" in feature['signal'] or "賣" in feature['signal'] else 0 
                            for feature in tech_features]
            
            # 繪製條形圖
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in feature_values]
            bars = ax1.bar(feature_names, feature_values, color=colors)
            
            # 添加數值標籤
            for bar, feature in zip(bars, tech_features):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., 
                        0.05 if height < 0 else height + 0.05,
                        feature['value'], 
                        ha='center', va='bottom', rotation=0, color='black')
            
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax1.set_title(f"{stock_id} 技術指標信號")
            ax1.set_ylim(-1.5, 1.5)
            ax1.set_ylabel("信號強度")
        else:
            ax1.text(0.5, 0.5, "無技術指標資料", horizontalalignment='center', verticalalignment='center')
        
        # 新聞情緒視覺化
        ax2 = axes[1]
        if news_features:
            # 準備情緒資料
            sentiment_data = {
                '正面新聞': news_features.get('positive_ratio', 0),
                '中性新聞': news_features.get('neutral_ratio', 0),
                '負面新聞': news_features.get('negative_ratio', 0),
                '情緒強度': abs(news_features.get('weighted_sentiment', 0)),
                '新聞趨勢': news_features.get('sentiment_trend', 0)
            }
            
            # 設定顏色
            colors = ['green', 'gray', 'red', 'blue', 'purple']
            
            # 繪製情緒資料
            pie_labels = list(sentiment_data.keys())[:3]  # 只選擇前三個做成餅圖
            pie_values = [sentiment_data[label] for label in pie_labels]
            
            if sum(pie_values) > 0:  # 確保有資料可以繪製
                ax2.pie(pie_values, labels=pie_labels, colors=colors[:3], autopct='%1.1f%%', startangle=90)
                ax2.axis('equal')  # 使餅圖為圓形
            
            # 添加情緒相關文字
            news_count = news_features.get('news_count', 0)
            sentiment_score = news_features.get('weighted_sentiment', 0)
            sentiment_text = f"股票: {stock_id}\n"
            sentiment_text += f"新聞數量: {news_count}\n"
            sentiment_text += f"情緒分數: {sentiment_score:.2f}\n"
            
            # 判斷情緒方向
            if sentiment_score > 0.2:
                sentiment_text += "整體情緒: 正面"
            elif sentiment_score < -0.2:
                sentiment_text += "整體情緒: 負面"
            else:
                sentiment_text += "整體情緒: 中性"
            
            ax2.set_title(sentiment_text)
        else:
            ax2.text(0.5, 0.5, "無新聞情緒資料", horizontalalignment='center', verticalalignment='center')
        
        # 綜合預測結果
        prediction_text = f"股票: {stock_id} 預測結果\n"
        prediction_text += f"預測方向: {'上漲' if prediction == 'up' else '下跌'}\n"
        prediction_text += f"信心指數: {probability:.2%}"
        
        plt.figtext(0.5, 0.01, prediction_text, ha='center', fontsize=14, 
                   bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 轉換為 base64
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"data:image/png;base64,{data}"
    
    except Exception as e:
        traceback.print_exc()
        return None

def generate_batch_prediction_chart(result):
    """生成批次預測結果圖表"""
    try:
        if result.get('status') != 'success' or 'results' not in result:
            return None
        
        results = result['results']
        
        # 轉換為DataFrame
        df = pd.DataFrame(results)
        
        # 過濾有效預測
        valid_df = df[df['prediction'].isin(['up', 'down'])]
        
        if valid_df.empty:
            return None
        
        # 計算上漲和下跌的股票數量
        up_stocks = valid_df[valid_df['prediction'] == 'up']
        down_stocks = valid_df[valid_df['prediction'] == 'down']
        
        # 創建圖表
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # 上漲/下跌比例餅圖
        labels = ['看多', '看空']
        sizes = [len(up_stocks), len(down_stocks)]
        colors = ['green', 'red']
        explode = (0.1, 0)  # 稍微突出上漲部分
        
        axs[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        axs[0].axis('equal')  # 使餅圖為圓形
        axs[0].set_title('上漲/下跌股票比例')
        
        # 信心分布直方圖
        # 轉換機率為數值
        valid_df['prob_value'] = valid_df['probability'].apply(
            lambda x: float(x) if isinstance(x, (int, float)) else 
            float(str(x).strip('%')) / 100 if isinstance(x, str) and '%' in str(x) else 0
        )
        
        # 上漲股票的信心分布
        if not up_stocks.empty:
            up_probs = up_stocks['prob_value']
            axs[1].hist(up_probs, bins=10, alpha=0.5, color='green', label='看多')
        
        # 下跌股票的信心分布
        if not down_stocks.empty:
            down_probs = down_stocks['prob_value']
            axs[1].hist(down_probs, bins=10, alpha=0.5, color='red', label='看空')
        
        axs[1].set_title('預測信心分布')
        axs[1].set_xlabel('信心水平')
        axs[1].set_ylabel('股票數量')
        axs[1].legend()
        
        plt.tight_layout()
        
        # 轉換為 base64
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"data:image/png;base64,{data}"
    
    except Exception as e:
        traceback.print_exc()
        return None

def generate_news_sentiment_chart(stock_id, news_result, sentiment_result):
    """生成新聞情緒圖表"""
    try:
        news_list = news_result.get('news', [])
        sentiment_summary = sentiment_result.get('sentiment', {})
        
        if not news_list:
            return None
        
        # 創建圖表
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # 情緒分布餅圖
        labels = ['正面', '中性', '負面']
        sizes = [
            sentiment_summary.get('positive_ratio', 0),
            sentiment_summary.get('neutral_ratio', 0),
            sentiment_summary.get('negative_ratio', 0)
        ]
        colors = ['green', 'gray', 'red']
        
        if sum(sizes) > 0:  # 確保有資料可以繪製
            axs[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axs[0].axis('equal')  # 使餅圖為圓形
            axs[0].set_title(f"{stock_id} 新聞情緒分布")
        else:
            axs[0].text(0.5, 0.5, "無新聞情緒資料", horizontalalignment='center', verticalalignment='center')
        
        # 新聞情緒隨時間變化圖
        df = pd.DataFrame(news_list)
        
        if 'publish_date' in df.columns and 'sentiment_score' in df.columns:
            # 轉換日期格式
            df['publish_date'] = pd.to_datetime(df['publish_date'])
            
            # 按日期排序
            df = df.sort_values('publish_date')
            
            # 繪製情緒分數變化
            axs[1].plot(df['publish_date'], df['sentiment_score'], 'g-', linewidth=1.5)
            
            # 標記正負區域
            axs[1].fill_between(
                df['publish_date'], 
                df['sentiment_score'], 
                0, 
                where=(df['sentiment_score'] >= 0),
                color='green', 
                alpha=0.3
            )
            axs[1].fill_between(
                df['publish_date'], 
                df['sentiment_score'], 
                0, 
                where=(df['sentiment_score'] <= 0),
                color='red', 
                alpha=0.3
            )
            
            # 設置標題和標籤
            axs[1].set_title(f"{stock_id} 新聞情緒走勢")
            axs[1].set_xlabel("日期")
            axs[1].set_ylabel("情緒分數")
            axs[1].set_ylim(-1, 1)
            axs[1].grid(True, linestyle='--', alpha=0.6)
            
            # 自動格式化日期
            fig.autofmt_xdate()
        else:
            axs[1].text(0.5, 0.5, "無法繪製情緒走勢", horizontalalignment='center', verticalalignment='center')
        
        # 添加摘要信息
        summary_text = f"股票: {stock_id}\n"
        summary_text += f"新聞數量: {sentiment_summary.get('news_count', 0)}\n"
        summary_text += f"平均情緒: {sentiment_summary.get('avg_sentiment', 0):.2f}"
        
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=14, 
                   bbox={"facecolor":"lightblue", "alpha":0.3, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 轉換為 base64
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"data:image/png;base64,{data}"
    
    except Exception as e:
        traceback.print_exc()
        return None

# 應用程式啟動
if __name__ == '__main__':
    # 檢查並建立靜態目錄
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # 檢查並建立模板目錄
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # 啟動服務
    port = 8000
    print(f"Web 應用啟動於 http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)