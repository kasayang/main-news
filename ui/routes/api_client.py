# api_client.py
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

class StockApiClient:
    """股票預測API客戶端"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """初始化API客戶端
        
        Args:
            base_url (str): API伺服器的基礎URL
        """
        self.base_url = base_url
    
    def predict_stock(self, stock_id: str, prediction_days: int = 1, model_type: str = "combined") -> Dict:
        """預測股票走勢
        
        Args:
            stock_id (str): 股票代碼
            prediction_days (int): 預測天數
            model_type (str): 模型類型 (combined, price_only, news_only)
        
        Returns:
            dict: 預測結果
        """
        url = f"{self.base_url}/api/predict"
        payload = {
            "stock_id": stock_id,
            "prediction_days": prediction_days,
            "model_type": model_type
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # 檢查HTTP狀態碼
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"預測請求失敗: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"錯誤詳情: {e.response.text}")
            return {"status": "error", "message": str(e)}
    
    def batch_predict(self, stock_ids: List[str], prediction_days: int = 1) -> Dict:
        """批次預測多支股票
        
        Args:
            stock_ids (list): 股票代碼列表
            prediction_days (int): 預測天數
        
        Returns:
            dict: 預測結果
        """
        url = f"{self.base_url}/api/batch_predict"
        payload = {
            "stock_ids": stock_ids,
            "prediction_days": prediction_days
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"批次預測請求失敗: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"錯誤詳情: {e.response.text}")
            return {"status": "error", "message": str(e)}
    
    def get_stock_news(self, stock_id: str, days: int = 30) -> Dict:
        """獲取股票相關新聞
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近天數
        
        Returns:
            dict: 新聞資料
        """
        url = f"{self.base_url}/api/news/{stock_id}?days={days}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"獲取新聞請求失敗: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_sentiment_summary(self, stock_id: str, days: int = 30) -> Dict:
        """獲取股票情緒摘要
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近天數
        
        Returns:
            dict: 情緒摘要
        """
        url = f"{self.base_url}/api/sentiment/{stock_id}?days={days}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"獲取情緒摘要請求失敗: {e}")
            return {"status": "error", "message": str(e)}
    
    def download_news(self, days: int = 7) -> Dict:
        """下載和分析新聞
        
        Args:
            days (int): 下載最近幾天的新聞
        
        Returns:
            dict: 下載結果
        """
        url = f"{self.base_url}/api/news/download"
        payload = {
            "days": days
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"下載新聞請求失敗: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_hot_topics(self, days: int = 7) -> Dict:
        """獲取熱門新聞主題
        
        Args:
            days (int): 最近天數
        
        Returns:
            dict: 熱門主題
        """
        url = f"{self.base_url}/api/news/topics?days={days}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"獲取熱門主題請求失敗: {e}")
            return {"status": "error", "message": str(e)}
    
    def train_model(self, stock_id: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, prediction_days: int = 1) -> Dict:
        """訓練預測模型
        
        Args:
            stock_id (str): 股票代碼
            start_date (str, optional): 起始日期，格式：YYYY-MM-DD
            end_date (str, optional): 結束日期，格式：YYYY-MM-DD
            prediction_days (int): 預測天數
        
        Returns:
            dict: 訓練結果
        """
        url = f"{self.base_url}/api/train"
        payload = {
            "stock_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
            "prediction_days": prediction_days
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"訓練模型請求失敗: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"錯誤詳情: {e.response.text}")
            return {"status": "error", "message": str(e)}
    
    def visualize_prediction_results(self, result: Dict) -> None:
        """視覺化預測結果
        
        Args:
            result (dict): 預測結果
        """
        if result.get('status') != 'success' or 'result' not in result:
            print("沒有可視覺化的預測結果")
            return
        
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
        plt.show()
    
    def visualize_batch_results(self, result: Dict) -> None:
        """視覺化批次預測結果
        
        Args:
            result (dict): 批次預測結果
        """
        if result.get('status') != 'success' or 'results' not in result:
            print("沒有可視覺化的批次預測結果")
            return
        
        results = result['results']
        
        # 轉換為DataFrame
        df = pd.DataFrame(results)
        
        # 過濾有效預測
        valid_df = df[df['prediction'].isin(['up', 'down'])]
        
        if valid_df.empty:
            print("沒有有效的預測結果")
            return
        
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
        plt.show()
        
        # 打印高信心預測
        high_confidence = valid_df[valid_df['prob_value'] > 0.7].sort_values('prob_value', ascending=False)
        
        if not high_confidence.empty:
            print("\n高信心預測股票:")
            for _, row in high_confidence.iterrows():
                print(f"{row['stock_id']}: {'上漲' if row['prediction'] == 'up' else '下跌'} (信心: {row['prob_value']:.2%})")
    
    def visualize_news_sentiment(self, stock_id: str, days: int = 30) -> None:
        """視覺化股票的新聞情緒
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近天數
        """
        # 取得新聞資料
        news_result = self.get_stock_news(stock_id, days)
        sentiment_result = self.get_sentiment_summary(stock_id, days)
        
        if news_result.get('status') != 'success' or sentiment_result.get('status') != 'success':
            print("無法取得新聞資料")
            return
        
        news_list = news_result.get('news', [])
        sentiment_summary = sentiment_result.get('sentiment', {})
        
        if not news_list:
            print(f"找不到 {stock_id} 的相關新聞")
            return
        
        # 轉換為DataFrame
        df = pd.DataFrame(news_list)
        
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
        plt.show()
        
        # 打印最近新聞標題
        print("\n最近新聞:")
        recent_news = sorted(news_list, key=lambda x: x.get('publish_date', ''), reverse=True)[:5]
        for news in recent_news:
            sentiment = news.get('sentiment_score', 0)
            sentiment_marker = '📈' if sentiment > 0.2 else '📉' if sentiment < -0.2 else '📊'
            print(f"{sentiment_marker} {news.get('publish_date', '')}: {news.get('title', '')}")


# 使用示例
if __name__ == "__main__":
    # 初始化客戶端
    client = StockApiClient("http://localhost:5000")
    
    # 示例1: 預測單一股票
    print("預測台積電(2330)走勢...")
    result = client.predict_stock("2330", prediction_days=1)
    if result.get('status') == 'success':
        print(f"預測結果: {result['result']['prediction']}, 機率: {result['result']['probability']:.2%}")
        client.visualize_prediction_results(result)
    
    # 示例2: 批次預測
    print("\n批次預測多檔股票...")
    batch_result = client.batch_predict(["2330", "2317", "2454"], prediction_days=1)
    if batch_result.get('status') == 'success':
        print(f"預測完成，共 {len(batch_result['results'])} 檔股票")
        client.visualize_batch_results(batch_result)
    
    # 示例3: 查看新聞情緒
    print("\n查看台積電(2330)的新聞情緒...")
    client.visualize_news_sentiment("2330", days=30)