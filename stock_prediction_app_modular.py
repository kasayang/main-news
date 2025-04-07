# stock_prediction_app_modular.py    main.py
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# 導入模組化的頁面
from ui.desktop.tabs.download_tab import DownloadTab
from ui.desktop.tabs.training_tab import TrainingTab
from ui.desktop.tabs.prediction_tab import PredictionTab
from ui.desktop.tabs.batch_prediction_tab import BatchPredictionTab
from ui.desktop.tabs.stock_filter_tab import StockFilterTab
from ui.desktop.tabs.news_tab import NewsTab  # 新增: 新聞頁面

class StockPredictionApp:
    def __init__(self, root):
        """初始化應用程式介面
        
        Args:
            root: Tkinter根視窗
        """
        self.root = root
        self.root.title("台股漲跌預測系統")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # 設置樣式
        self.setup_styles()
        
        # 建立UI
        self.create_ui()
    
    def setup_styles(self):
        """設置Tkinter樣式"""
        style = ttk.Style()
        
        # 檢查當前系統，設置合適的主題
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
        
        # 設置強調按鈕樣式
        style.configure("Accent.TButton", font=("Arial", 11, "bold"))
        
        # 設置標籤框架樣式
        style.configure("TLabelframe", borderwidth=2)
        style.configure("TLabelframe.Label", font=("Arial", 12, "bold"))
        
        # 設置表格樣式
        style.configure("Treeview", font=("Arial", 10))
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))
        
        # 設置頁籤樣式
        style.configure("TNotebook.Tab", font=("Arial", 11))
    
    def create_ui(self):
        """建立使用者介面"""
        # 建立Tab控制項
        self.tab_control = ttk.Notebook(self.root)
        
        # 資料下載頁面
        self.tab_download = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_download, text="資料下載")
        
        # 模型訓練頁面
        self.tab_train = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_train, text="模型訓練")
        
        # 預測頁面
        self.tab_predict = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_predict, text="股票預測")
        
        # 批次預測頁面
        self.tab_batch = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_batch, text="批次預測")
        
        # 股票篩選頁面
        self.tab_filter = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_filter, text="股票篩選")
        
        # 新聞分析頁面
        self.tab_news = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_news, text="新聞分析")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # 建立狀態列
        self.status_var = tk.StringVar()
        self.status_var.set("系統就緒")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 初始化各頁面
        self.download_tab = DownloadTab(self.tab_download, self.root)
        self.training_tab = TrainingTab(self.tab_train, self.root)
        self.prediction_tab = PredictionTab(self.tab_predict, self.root)
        self.batch_prediction_tab = BatchPredictionTab(self.tab_batch, self.root)
        self.stock_filter_tab = StockFilterTab(self.tab_filter, self.root)
        self.news_tab = NewsTab(self.tab_news, self.root)  # 新增新聞分析頁面
        
        # 設定頁面間資料共享
        self.setup_data_sharing()
    
    def setup_data_sharing(self):
        """設置頁面間的資料共享機制"""
        # 定義共用的資料庫路徑和模型目錄
        db_path = "tw_stock_data.db"
        model_dir = "models"
        
        # 更新各頁面的資料庫路徑和模型目錄
        for tab in [self.download_tab, self.training_tab, self.prediction_tab, 
                   self.batch_prediction_tab, self.stock_filter_tab, self.news_tab]:
            if hasattr(tab, 'db_path'):
                tab.db_path = db_path
            
            if hasattr(tab, 'model_dir'):
                tab.model_dir = model_dir
        
        # 設定從股票篩選器到預測頁面的跳轉功能
        if hasattr(self.stock_filter_tab, 'filter_result_tree'):
            self.stock_filter_tab.filter_result_tree.bind("<Double-1>", self.jump_to_prediction)
    
    def jump_to_prediction(self, event):
        """從篩選結果跳轉到預測頁面"""
        # 獲取選中的股票
        selected_items = self.stock_filter_tab.filter_result_tree.selection()
        if not selected_items:
            return
            
        # 獲取股票代碼
        item = selected_items[0]
        values = self.stock_filter_tab.filter_result_tree.item(item, "values")
        stock_id = values[0]  # 假設第一列是股票代碼
        
        # 切換到預測頁面
        self.tab_control.select(self.tab_predict)
        
        # 更新預測頁面的股票代碼
        if hasattr(self.prediction_tab, 'predict_stock_id_var'):
            self.prediction_tab.predict_stock_id_var.set(stock_id)
            
            # 觸發預測
            if hasattr(self.prediction_tab, 'predict_stock'):
                self.prediction_tab.predict_stock()
    
    def integrate_news_into_prediction(self):
        """整合新聞分析到股價預測中"""
        # 載入新聞分析模型
        from models.news_model import NewsModel
        news_model = NewsModel(database_path=self.download_tab.db_path, model_dir=self.training_tab.model_dir)
        
        # 將新聞模型加入到預測頁面
        self.prediction_tab.news_model = news_model
        
        # 擴展預測功能，以使用新聞情緒
        original_predict_thread = self.prediction_tab.predict_stock_thread
        
        def enhanced_predict_thread(stock_id, prediction_days):
            """增強版的預測執行緒，結合新聞情緒分析"""
            try:
                # 首先執行原始預測
                original_predict_thread(stock_id, prediction_days)
                
                # 獲取新聞情緒分析
                if hasattr(self.prediction_tab, 'news_model'):
                    news_features = self.prediction_tab.news_model.get_sentiment_features(stock_id)
                    
                    # 更新預測結果顯示，加入新聞分析信息
                    original_result = self.prediction_tab.predict_result_var.get()
                    
                    # 新聞情緒評分
                    sentiment_score = news_features['weighted_sentiment']
                    sentiment_text = ""
                    if sentiment_score > 0.2:
                        sentiment_text = f"\n新聞情緒: 正面 (+{sentiment_score:.2f})"
                    elif sentiment_score < -0.2:
                        sentiment_text = f"\n新聞情緒: 負面 ({sentiment_score:.2f})"
                    else:
                        sentiment_text = f"\n新聞情緒: 中性 ({sentiment_score:.2f})"
                    
                    # 新聞影響
                    news_trend = news_features['sentiment_trend']
                    if news_trend > 0.01:
                        trend_text = f"趨勢上升"
                    elif news_trend < -0.01:
                        trend_text = f"趨勢下降"
                    else:
                        trend_text = f"趨勢穩定"
                    
                    # 更新顯示
                    enhanced_result = original_result + sentiment_text + f"\n新聞趨勢: {trend_text}"
                    self.prediction_tab.update_predict_result(enhanced_result)
            
            except Exception as e:
                error_msg = f"整合新聞分析時發生錯誤: {str(e)}"
                self.prediction_tab.update_predict_result(f"錯誤: {error_msg}")
        
        # 替換原始預測執行緒
        self.prediction_tab.predict_stock_thread = enhanced_predict_thread

# 應用程式進入點
if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    
    # 整合新聞分析到股價預測中
    app.integrate_news_into_prediction()
    
    root.mainloop()