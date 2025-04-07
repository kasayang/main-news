# ui/desktop/tabs/combined_prediction_tab.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd

# 導入模型
from models.combined_model import CombinedModel

class CombinedPredictionTab:
    def __init__(self, parent, root):
        """初始化綜合預測頁面
        
        Args:
            parent: 父容器
            root: 主視窗
        """
        self.parent = parent
        self.root = root
        self.predictor = None
        self.model_dir = "models"
        self.db_path = "tw_stock_data.db"
        self.current_figure = None
        
        # 建立UI
        self.setup_combined_predict_tab()
    
    def setup_combined_predict_tab(self):
        """設置綜合預測頁面"""
        frame = ttk.LabelFrame(self.parent, text="綜合股票漲跌預測 (技術分析 + 新聞分析)")
        frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # 分成左右兩區
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill="both", padx=5, pady=5, expand=False)
        
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill="both", padx=5, pady=5, expand=True)
        
        # 左側控制區
        control_frame = ttk.LabelFrame(left_frame, text="選項與設定")
        control_frame.pack(fill="both", padx=5, pady=5, expand=True)
        
        # 股票選擇
        stock_frame = ttk.Frame(control_frame)
        stock_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(stock_frame, text="股票代碼:").pack(side=tk.LEFT, padx=5)
        self.predict_stock_id_var = tk.StringVar()
        ttk.Entry(stock_frame, textvariable=self.predict_stock_id_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # 預測天數
        pred_frame = ttk.Frame(control_frame)
        pred_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(pred_frame, text="預測未來幾天:").pack(side=tk.LEFT, padx=5)
        self.predict_days_var = tk.IntVar(value=1)
        ttk.Combobox(
            pred_frame, 
            textvariable=self.predict_days_var, 
            values=[1, 3, 5, 10], 
            width=5,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        # 模型選擇
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(model_frame, text="預測模型:").pack(side=tk.LEFT, padx=5)
        self.model_type_var = tk.StringVar(value="combined")
        ttk.Combobox(
            model_frame, 
            textvariable=self.model_type_var, 
            values=["combined", "price_only", "news_only"], 
            width=10,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        # 預測按鈕
        predict_btn = ttk.Button(
            control_frame, 
            text="進行預測", 
            command=self.predict_stock,
            style="Accent.TButton"
        )
        predict_btn.pack(fill="x", padx=5, pady=10, ipady=5)
        
        # 批次預測區域
        batch_frame = ttk.LabelFrame(left_frame, text="批次預測")
        batch_frame.pack(fill="both", padx=5, pady=5, expand=True)
        
        # 批次股票列表
        ttk.Label(batch_frame, text="股票代碼清單:").pack(anchor=tk.W, padx=5, pady=2)
        self.batch_stocks_var = tk.StringVar()
        ttk.Entry(batch_frame, textvariable=self.batch_stocks_var, width=20).pack(fill="x", padx=5, pady=2)
        ttk.Label(batch_frame, text="(以逗號分隔，如: 2330,2454)").pack(anchor=tk.W, padx=5, pady=2)
        
        # 批次預測按鈕
        ttk.Button(
            batch_frame,
            text="批次預測",
            command=self.batch_predict
        ).pack(fill="x", padx=5, pady=5)
        
        # 右側資訊區域
        # 上方：預測結果
        result_frame = ttk.LabelFrame(right_frame, text="預測結果")
        result_frame.pack(fill="x", padx=5, pady=5)
        
        # 預測結果區域
        result_container = ttk.Frame(result_frame)
        result_container.pack(fill="both", padx=5, pady=5, expand=True)
        
        # 左側：技術分析結果
        tech_frame = ttk.LabelFrame(result_container, text="技術分析")
        tech_frame.pack(side=tk.LEFT, fill="both", padx=5, pady=5, expand=True)
        
        self.tech_result_var = tk.StringVar(value="尚未預測")
        ttk.Label(tech_frame, textvariable=self.tech_result_var, font=("Arial", 11), wraplength=250).pack(padx=10, pady=10)
        
        # 右側：新聞分析結果
        news_frame = ttk.LabelFrame(result_container, text="新聞分析")
        news_frame.pack(side=tk.LEFT, fill="both", padx=5, pady=5, expand=True)
        
        self.news_result_var = tk.StringVar(value="尚未預測")
        ttk.Label(news_frame, textvariable=self.news_result_var, font=("Arial", 11), wraplength=250).pack(padx=10, pady=10)
        
        # 中間：綜合結果
        combined_frame = ttk.LabelFrame(result_frame, text="綜合預測結果")
        combined_frame.pack(fill="x", padx=5, pady=5)
        
        self.combined_result_var = tk.StringVar(value="尚未預測")
        ttk.Label(combined_frame, textvariable=self.combined_result_var, font=("Arial", 12, "bold"), wraplength=500).pack(padx=10, pady=10)
        
        # 下方：圖表區域
        chart_frame = ttk.LabelFrame(right_frame, text="股價與情緒走勢")
        chart_frame.pack(fill="both", padx=5, pady=5, expand=True)
        
        # 圖表容器
        self.chart_container = ttk.Frame(chart_frame)
        self.chart_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 底部：批次預測結果
        batch_result_frame = ttk.LabelFrame(right_frame, text="批次預測結果")
        batch_result_frame.pack(fill="both", padx=5, pady=5)
        
        # 批次預測結果表格
        columns = ("股票代碼", "目標日期", "預測方向", "預測機率", "技術信號", "新聞信號", "現價")
        self.batch_result_tree = ttk.Treeview(batch_result_frame, columns=columns, show="headings", height=5)
        
        # 設置列標題
        for col in columns:
            self.batch_result_tree.heading(col, text=col)
            if col == "股票代碼":
                self.batch_result_tree.column(col, width=80, anchor="center")
            elif col in ["預測方向", "技術信號", "新聞信號"]:
                self.batch_result_tree.column(col, width=80, anchor="center")
            else:
                self.batch_result_tree.column(col, width=100, anchor="center")
        
        # 表格事件處理
        self.batch_result_tree.bind("<Double-1>", self.on_batch_result_double_click)
        
        # 添加滾動條
        scrollbar = ttk.Scrollbar(batch_result_frame, orient="horizontal", command=self.batch_result_tree.xview)
        self.batch_result_tree.configure(xscrollcommand=scrollbar.set)
        scrollbar.pack(side="bottom", fill="x")
        self.batch_result_tree.pack(fill="both", expand=True)
    
    def predict_stock(self):
        """進行單一股票的綜合預測"""
        stock_id = self.predict_stock_id_var.get().strip()
        if not stock_id:
            messagebox.showerror("錯誤", "請輸入股票代碼")
            return
        
        prediction_days = self.predict_days_var.get()
        model_type = self.model_type_var.get()
        
        # 更新狀態
        if hasattr(self.root, 'status_var'):
            self.root.status_var.set(f"正在分析 {stock_id} 的技術指標和新聞情緒...")
        
        # 清空之前的預測結果
        self.tech_result_var.set("分析中...")
        self.news_result_var.set("分析中...")
        self.combined_result_var.set("分析中...")
        
        # 在背景執行緒中進行預測
        threading.Thread(
            target=self.predict_stock_thread,
            args=(stock_id, prediction_days, model_type)
        ).start()
    
    def predict_stock_thread(self, stock_id, prediction_days, model_type):
        """在背景執行緒中進行預測分析"""
        try:
            # 初始化模型
            if self.predictor is None:
                self.predictor = CombinedModel(database_path=self.db_path, model_dir=self.model_dir)
            
            # 根據模型類型選擇不同的預測方法
            if model_type == "price_only":
                # 只使用價格模型
                result = self.predictor.price_predictor.predict_price_movement(stock_id, prediction_days=prediction_days)
                tech_only = True
                news_only = False
            elif model_type == "news_only":
                # 只使用新聞模型
                direction, confidence = self.predictor.news_model.predict_price_movement_from_news(stock_id)
                result = {
                    'stock_id': stock_id,
                    'prediction_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                    'target_date': (datetime.datetime.now() + datetime.timedelta(days=prediction_days)).strftime('%Y-%m-%d'),
                    'prediction_days': prediction_days,
                    'prediction': 'up' if direction > 0 else 'down',
                    'probability': confidence,
                    'model_type': "news_only"
                }
                tech_only = False
                news_only = True
            else:
                # 使用綜合模型
                result = self.predictor.predict_price_movement(stock_id, prediction_days=prediction_days)
                tech_only = False
                news_only = False
            
            if result:
                # 更新預測結果顯示
                # 1. 技術分析結果
                tech_text = ""
                if not news_only:
                    tech_features = self.get_tech_features(stock_id)
                    tech_signal = "看多" if result['prediction'] == 'up' else "看空"
                    tech_text = f"技術分析結果: {tech_signal}\n"
                    tech_text += f"預測機率: {result['probability']:.2%}\n\n"
                    
                    # 添加技術指標信息
                    if tech_features:
                        tech_text += "關鍵技術指標:\n"
                        for name, value, signal in tech_features:
                            tech_text += f"{name}: {value} ({signal})\n"
                self.update_tech_result(tech_text)
                
                # 2. 新聞分析結果
                news_text = ""
                if not tech_only:
                    news_features = self.predictor.news_model.get_sentiment_features(stock_id)
                    news_signal = "利多" if news_features['weighted_sentiment'] > 0 else "利空" if news_features['weighted_sentiment'] < 0 else "中性"
                    news_text = f"新聞情緒: {news_signal}\n"
                    news_text += f"情緒強度: {abs(news_features['weighted_sentiment']):.2f}\n"
                    news_text += f"新聞數量: {news_features['news_count']}\n\n"
                    
                    # 添加情緒分佈
                    if news_features['news_count'] > 0:
                        news_text += f"正面新聞: {news_features['positive_ratio']:.1%}\n"
                        news_text += f"負面新聞: {news_features['negative_ratio']:.1%}\n"
                        news_text += f"中性新聞: {news_features['neutral_ratio']:.1%}\n"
                    
                    # 添加最近變化與趨勢
                    trend = news_features['sentiment_trend']
                    if trend > 0.01:
                        trend_text = "上升"
                    elif trend < -0.01:
                        trend_text = "下降"
                    else:
                        trend_text = "穩定"
                    news_text += f"\n新聞趨勢: {trend_text}"
                self.update_news_result(news_text)
                
                # 3. 綜合預測結果
                combined_text = f"股票: {stock_id}\n"
                combined_text += f"現價: {result['current_price']}\n"
                combined_text += f"預測區間: {result['prediction_date']} 至 {result['target_date']}\n"
                combined_text += f"綜合預測: {result['target_date']} 將會 {result['prediction']}\n"
                combined_text += f"預測信心: {result['probability']:.2%}\n"
                
                if 'model_type' in result:
                    model_text = ""
                    if result['model_type'] == "combined":
                        model_text = "綜合模型 (技術+新聞)"
                    elif result['model_type'] == "price_only":
                        model_text = "僅技術分析"
                    elif result['model_type'] == "news_only":
                        model_text = "僅新聞分析"
                    elif result['model_type'] == "price_with_news":
                        model_text = "技術分析+新聞權重調整"
                    
                    combined_text += f"模型類型: {model_text}"
                
                self.update_combined_result(combined_text)
                
                # 4. 繪製圖表
                self.plot_combined_chart(stock_id, prediction_days)
                
                # 更新狀態
                if hasattr(self.root, 'status_var'):
                    self.root.status_var.set(f"{stock_id} 綜合預測完成")
            else:
                # 預測失敗
                self.update_tech_result("技術分析失敗，無法獲取足夠數據")
                self.update_news_result("新聞分析失敗，無法獲取足夠數據")
                self.update_combined_result("綜合預測失敗，請檢查股票代碼或嘗試訓練模型")
                
                if hasattr(self.root, 'status_var'):
                    self.root.status_var.set(f"{stock_id} 預測失敗")
        except Exception as e:
            error_msg = f"預測分析時發生錯誤: {str(e)}"
            self.update_tech_result("分析錯誤")
            self.update_news_result("分析錯誤")
            self.update_combined_result(f"錯誤: {error_msg}")
            
            if hasattr(self.root, 'status_var'):
                self.root.status_var.set(error_msg)
    
    def get_tech_features(self, stock_id):
        """獲取關鍵技術指標信息"""
        try:
            conn = self.predictor.conn
            
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
                tech_features.append(("5日/10日均線", f"{ma5:.2f}/{ma10:.2f}", ma_signal))
            
            if pd.notna(close) and pd.notna(ma20):
                if close > ma20:
                    price_signal = "偏多"
                else:
                    price_signal = "偏空"
                tech_features.append(("價格/20日均線", f"{close:.2f}/{ma20:.2f}", price_signal))
            
            # RSI值
            rsi = df['rsi'].iloc[0]
            if pd.notna(rsi):
                if rsi > 70:
                    rsi_signal = "超買"
                elif rsi < 30:
                    rsi_signal = "超賣"
                else:
                    rsi_signal = "中性"
                tech_features.append(("RSI", f"{rsi:.2f}", rsi_signal))
            
            # MACD
            macd = df['macd'].iloc[0]
            if pd.notna(macd):
                if macd > 0:
                    macd_signal = "偏多"
                else:
                    macd_signal = "偏空"
                tech_features.append(("MACD", f"{macd:.4f}", macd_signal))
            
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
                tech_features.append(("成交量比例", f"{vol_ratio:.2f}", vol_signal))
            
            return tech_features
        
        except Exception as e:
            print(f"獲取技術指標時發生錯誤: {str(e)}")
            return []
    
    def update_tech_result(self, text):
        """更新技術分析結果"""
        self.root.after(0, lambda: self.tech_result_var.set(text))
    
    def update_news_result(self, text):
        """更新新聞分析結果"""
        self.root.after(0, lambda: self.news_result_var.set(text))
    
    def update_combined_result(self, text):
        """更新綜合預測結果"""
        self.root.after(0, lambda: self.combined_result_var.set(text))
    
    def plot_combined_chart(self, stock_id, prediction_days):
        """繪製綜合圖表，包含股價走勢和新聞情緒"""
        try:
            # 清除舊圖表
            for widget in self.chart_container.winfo_children():
                widget.destroy()
            
            # 獲取股價數據
            conn = self.predictor.conn
            days = 120  # 顯示過去120天數據
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 價格數據
            price_query = f"""
            SELECT p.date, p.close, 
                   m5.close as ma5, 
                   m10.close as ma10, 
                   m20.close as ma20
            FROM price_close p
            LEFT JOIN (
                SELECT date, stock_id, AVG(close) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as close
                FROM price_close
                WHERE stock_id = '{stock_id}'
            ) m5 ON p.date = m5.date AND p.stock_id = m5.stock_id
            LEFT JOIN (
                SELECT date, stock_id, AVG(close) OVER (ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as close
                FROM price_close
                WHERE stock_id = '{stock_id}'
            ) m10 ON p.date = m10.date AND p.stock_id = m10.stock_id
            LEFT JOIN (
                SELECT date, stock_id, AVG(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as close
                FROM price_close
                WHERE stock_id = '{stock_id}'
            ) m20 ON p.date = m20.date AND p.stock_id = m20.stock_id
            WHERE p.stock_id = '{stock_id}'
            AND p.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY p.date
            """
            
            price_df = pd.read_sql(price_query, conn)
            
            # 成交量數據
            volume_query = f"""
            SELECT date, volume
            FROM price_volume
            WHERE stock_id = '{stock_id}'
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
            """
            
            volume_df = pd.read_sql(volume_query, conn)
            
            # 新聞情緒數據
            sentiment_query = f"""
            SELECT n.publish_date as date, AVG(s.sentiment_score * r.relevance) as sentiment_score, COUNT(*) as count
            FROM stock_news n
            JOIN news_stock_relation r ON n.id = r.news_id
            JOIN news_sentiment s ON n.id = s.news_id
            WHERE r.stock_id = '{stock_id}'
            AND n.publish_date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY n.publish_date
            ORDER BY n.publish_date
            """
            
            sentiment_df = pd.read_sql(sentiment_query, conn)
            
            # 轉換日期格式
            price_df['date'] = pd.to_datetime(price_df['date'])
            volume_df['date'] = pd.to_datetime(volume_df['date'])
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            
            # 創建子圖
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
            
            # 股價圖 (top)
            ax1 = fig.add_subplot(gs[0])
            
            # 繪製股價和均線
            ax1.plot(price_df['date'], price_df['close'], 'b-', linewidth=1.5, label='收盤價')
            if 'ma5' in price_df.columns:
                ax1.plot(price_df['date'], price_df['ma5'], 'r--', linewidth=1, label='5日均線')
            if 'ma10' in price_df.columns:
                ax1.plot(price_df['date'], price_df['ma10'], 'g--', linewidth=1, label='10日均線')
            if 'ma20' in price_df.columns:
                ax1.plot(price_df['date'], price_df['ma20'], 'c--', linewidth=1, label='20日均線')
            
            # 添加預測結果標記
            latest_date = price_df['date'].max()
            latest_price = price_df.loc[price_df['date'] == latest_date, 'close'].values[0]
            
            # 預測方向
            prediction = self.combined_result_var.get()
            if "上漲" in prediction:
                prediction_dir = "上漲"
                arrow_color = 'red'
            elif "下跌" in prediction:
                prediction_dir = "下跌"
                arrow_color = 'green'
            else:
                prediction_dir = None
            
            # 添加預測箭頭
            if prediction_dir:
                target_date = latest_date + pd.Timedelta(days=prediction_days)
                
                # 估算目標價格（簡化處理，實際應使用模型預測結果）
                if prediction_dir == "上漲":
                    target_price = latest_price * 1.05  # 假設上漲5%
                else:
                    target_price = latest_price * 0.95  # 假設下跌5%
                
                # 繪製預測箭頭
                ax1.annotate(
                    '',
                    xy=(target_date, target_price),
                    xytext=(latest_date, latest_price),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2),
                    annotation_clip=False
                )
                
                # 添加預測文字說明
                ax1.text(
                    latest_date + (target_date - latest_date) * 0.5,
                    (latest_price + target_price) * 0.5,
                    f"預測{prediction_dir}",
                    color=arrow_color,
                    fontweight='bold',
                    ha='center'
                )
            
            # 設置標題和標籤
            ax1.set_title(f"{stock_id} 股價走勢與預測", fontsize=14)
            ax1.set_ylabel("價格", fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper left')
            
            # 成交量圖 (middle)
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            
            if not volume_df.empty:
                ax2.bar(volume_df['date'], volume_df['volume'], color='blue', alpha=0.6)
                ax2.set_ylabel("成交量", fontsize=12)
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 情緒圖 (bottom)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            
            if not sentiment_df.empty:
                # 繪製情緒分數
                ax3.plot(sentiment_df['date'], sentiment_df['sentiment_score'], 'g-', linewidth=1.5)
                
                # 填充正負區域
                ax3.fill_between(
                    sentiment_df['date'], 
                    sentiment_df['sentiment_score'], 
                    0, 
                    where=(sentiment_df['sentiment_score'] >= 0),
                    color='green', 
                    alpha=0.3
                )
                ax3.fill_between(
                    sentiment_df['date'], 
                    sentiment_df['sentiment_score'], 
                    0, 
                    where=(sentiment_df['sentiment_score'] <= 0),
                    color='red', 
                    alpha=0.3
                )
                
                # 繪製新聞數量作為點大小
                sizes = sentiment_df['count'] * 20  # 放大點的大小以便於觀察
                scatter = ax3.scatter(
                    sentiment_df['date'], 
                    sentiment_df['sentiment_score'],
                    s=sizes,
                    c=sentiment_df['sentiment_score'],
                    cmap='RdYlGn',
                    vmin=-1,
                    vmax=1,
                    alpha=0.7
                )
                
                # 添加 0 線
                ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                
                ax3.set_ylabel("新聞情緒", fontsize=12)
                ax3.set_ylim(-1, 1)
                ax3.grid(True, linestyle='--', alpha=0.7)
            
            # 調整 x 軸格式
            plt.gcf().autofmt_xdate()
            
            # 添加圖例
            fig.colorbar(scatter, ax=ax3, label='情緒分數')
            
            # 調整佈局
            plt.tight_layout()
            
            # 在 UI 中顯示圖表
            canvas = FigureCanvasTkAgg(fig, self.chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            # 添加工具列
            toolbar_frame = ttk.Frame(self.chart_container)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            # 保存當前圖表
            self.current_figure = fig
            
        except Exception as e:
            print(f"繪製圖表時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def batch_predict(self):
        """批次預測多檔股票"""
        # 獲取股票列表
        stocks_text = self.batch_stocks_var.get().strip()
        if not stocks_text:
            messagebox.showerror("錯誤", "請輸入股票代碼清單")
            return
        
        # 解析股票代碼
        stock_ids = [s.strip() for s in stocks_text.split(',')]
        if not stock_ids:
            messagebox.showerror("錯誤", "無法解析股票代碼清單")
            return
        
        prediction_days = self.predict_days_var.get()
        
        # 更新狀態
        if hasattr(self.root, 'status_var'):
            self.root.status_var.set(f"正在批次預測 {len(stock_ids)} 檔股票...")
        
        # 清空結果表格
        for item in self.batch_result_tree.get_children():
            self.batch_result_tree.delete(item)
        
        # 在背景執行緒中進行批次預測
        threading.Thread(
            target=self.batch_predict_thread,
            args=(stock_ids, prediction_days)
        ).start()
    
    def batch_predict_thread(self, stock_ids, prediction_days):
        """在背景執行緒中批次預測"""
        try:
            # 初始化模型
            if self.predictor is None:
                self.predictor = CombinedModel(database_path=self.db_path, model_dir=self.model_dir)
            
            # 批次預測結果
            results = []
            
            # 進行預測
            for stock_id in stock_ids:
                try:
                    # 綜合預測
                    prediction = self.predictor.predict_price_movement(stock_id, prediction_days=prediction_days)
                    
                    if prediction:
                        # 獲取新聞情緒信號
                        try:
                            news_features = self.predictor.news_model.get_sentiment_features(stock_id)
                            sentiment_score = news_features['weighted_sentiment']
                            if sentiment_score > 0.2:
                                news_signal = "利多"
                            elif sentiment_score < -0.2:
                                news_signal = "利空"
                            else:
                                news_signal = "中性"
                        except:
                            news_signal = "N/A"
                        
                        # 獲取技術分析信號
                        tech_features = self.get_tech_features(stock_id)
                        if tech_features:
                            up_signals = sum(1 for _, _, signal in tech_features if "多" in signal or "買" in signal)
                            down_signals = sum(1 for _, _, signal in tech_features if "空" in signal or "賣" in signal)
                            if up_signals > down_signals:
                                tech_signal = "看多"
                            elif down_signals > up_signals:
                                tech_signal = "看空"
                            else:
                                tech_signal = "中性"
                        else:
                            tech_signal = "N/A"
                        
                        # 添加到結果列表
                        self.add_batch_result(
                            stock_id,
                            prediction['target_date'],
                            prediction['prediction'],
                            prediction['probability'],
                            tech_signal,
                            news_signal,
                            prediction['current_price']
                        )
                        
                        results.append(prediction)
                except Exception as stock_error:
                    self.logger.error(f"預測 {stock_id} 時發生錯誤: {stock_error}")
                    
                    # 添加錯誤結果
                    self.add_batch_result(
                        stock_id,
                        "-",
                        "錯誤",
                        0,
                        "N/A",
                        "N/A",
                        0
                    )
            
            # 更新狀態
            if hasattr(self.root, 'status_var'):
                self.root.status_var.set(f"完成 {len(stock_ids)} 檔股票的批次預測")
        
        except Exception as e:
            error_msg = f"批次預測時發生錯誤: {str(e)}"
            if hasattr(self.root, 'status_var'):
                self.root.status_var.set(error_msg)
            messagebox.showerror("錯誤", error_msg)
    
    def add_batch_result(self, stock_id, target_date, prediction, probability, tech_signal, news_signal, current_price):
        """添加批次預測結果到表格"""
        # 轉換概率為百分比
        if isinstance(probability, (int, float)):
            prob_str = f"{probability:.2%}"
        else:
            prob_str = str(probability)
        
        # 轉換價格
        if isinstance(current_price, (int, float)):
            price_str = f"{current_price:.2f}"
        else:
            price_str = str(current_price)
        
        # 添加到表格
        self.root.after(0, lambda: self.batch_result_tree.insert(
            "", "end",
            values=(stock_id, target_date, prediction, prob_str, tech_signal, news_signal, price_str)
        ))
    
    def on_batch_result_double_click(self, event):
        """點擊批次預測結果的處理事件"""
        selection = self.batch_result_tree.selection()
        if not selection:
            return
        
        # 獲取選中的股票
        item = selection[0]
        values = self.batch_result_tree.item(item, "values")
        stock_id = values[0]
        
        # 更新股票代碼
        self.predict_stock_id_var.set(stock_id)
        
        # 進行單一股票詳細預測
        self.predict_stock()
        