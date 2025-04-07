# analyzers/news_analyzer.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re
from typing import Dict, List, Tuple, Optional
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt

class NewsAnalyzer:
    """新聞分析器，負責分析和處理新聞資料"""
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """初始化新聞分析器
        
        Args:
            database_path (str): 資料庫路徑
        """
        self.database_path = database_path
        self.conn = sqlite3.connect(database_path)
        
        # 設置日誌
        self.logger = logging.getLogger("NewsAnalyzer")
        
        # 載入自定義詞典（如果有）
        try:
            jieba.load_userdict("dict/financial_terms.txt")
        except:
            self.logger.info("未找到自定義財經詞典，使用預設詞典")
    
    def analyze_news_sentiment(self, news_id: Optional[int] = None) -> int:
        """分析新聞情緒
        
        Args:
            news_id (int, optional): 特定新聞ID，如果為None則分析全部未分析的新聞
        
        Returns:
            int: 處理的新聞數量
        """
        # 情感詞典
        positive_words = [
            "上漲", "成長", "獲利", "增加", "好", "看好", "強", "正面", "樂觀", "利多", 
            "漲", "增", "升", "盈利", "成功", "豐厚", "熱絡", "穩健", "卓越", "傑出",
            "佳", "優", "耀眼", "領先", "積極", "優化", "利", "加值", "豐盛", "順暢",
            "賺", "回升", "提升", "興隆", "突破", "創新高", "高成長", "創高", "創收", "超額"
        ]
        
        negative_words = [
            "下跌", "虧損", "衰退", "減少", "差", "看空", "弱", "負面", "悲觀", "利空", 
            "跌", "減", "降", "虧", "失敗", "虧累", "冷清", "不穩", "惡化", "頹勢",
            "壞", "劣", "黯淡", "落後", "消極", "惡化", "損", "縮水", "短缺", "受阻",
            "賠", "下滑", "走低", "蕭條", "受創", "創新低", "低迷", "創低", "疲弱", "不及"
        ]
        
        # 查詢要分析的新聞
        if news_id:
            query = "SELECT id, title, content FROM stock_news WHERE id = ?"
            params = (news_id,)
        else:
            # 查詢所有尚未分析情緒的新聞
            query = """
            SELECT n.id, n.title, n.content 
            FROM stock_news n
            LEFT JOIN news_sentiment s ON n.id = s.news_id
            WHERE s.news_id IS NULL
            """
            params = None
        
        # 執行查詢
        df = pd.read_sql(query, self.conn, params=params)
        
        if df.empty:
            self.logger.info("未找到需要分析的新聞")
            return 0
        
        self.logger.info(f"開始分析 {len(df)} 篇新聞情緒")
        
        # 情緒分析結果
        results = []
        
        # 逐篇分析
        for _, row in df.iterrows():
            news_id = row['id']
            title = row['title']
            content = row['content'] if pd.notna(row['content']) else ""
            
            # 合併標題和內容，標題權重更高
            text = title + " " + title + " " + content
            
            # 分詞
            words = [word for word in jieba.cut(text) if len(word) > 1]
            
            # 統計正負面詞彙
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            # 提取關鍵字
            keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=True)
            
            # 計算情緒分數
            total = positive_count + negative_count
            if total > 0:
                sentiment_score = (positive_count - negative_count) / total
                positive_score = positive_count / total
                negative_score = negative_count / total
            else:
                sentiment_score = 0
                positive_score = 0.5
                negative_score = 0.5
            
            neutral_score = 1 - abs(sentiment_score)
            
            # 生成摘要（簡單取前100個字符）
            summary = title
            if len(content) > 100:
                summary = content[:100] + "..."
            
            # 整理關鍵字
            keywords_text = ",".join([f"{word}" for word, weight in keywords])
            
            # 添加到結果列表
            results.append({
                'news_id': news_id,
                'sentiment_score': sentiment_score,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'neutral_score': neutral_score,
                'keywords': keywords_text,
                'summary': summary
            })
        
        # 批量保存分析結果
        cursor = self.conn.cursor()
        for result in results:
            cursor.execute("""
            INSERT OR REPLACE INTO news_sentiment 
            (news_id, sentiment_score, positive_score, negative_score, neutral_score, keywords, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result['news_id'],
                result['sentiment_score'],
                result['positive_score'],
                result['negative_score'],
                result['neutral_score'],
                result['keywords'],
                result['summary']
            ))
        
        self.conn.commit()
        self.logger.info(f"成功分析 {len(results)} 篇新聞情緒")
        
        return len(results)
    
    def analyze_related_stocks(self, news_id: Optional[int] = None) -> int:
        """分析新聞中提到的相關股票
        
        Args:
            news_id (int, optional): 特定新聞ID，如果為None則分析全部未分析的新聞
        
        Returns:
            int: 處理的新聞數量
        """
        # 查詢要分析的新聞
        if news_id:
            query = "SELECT id, title, content FROM stock_news WHERE id = ?"
            params = (news_id,)
        else:
            # 查詢所有尚未分析相關股票的新聞
            query = """
            SELECT n.id, n.title, n.content 
            FROM stock_news n
            LEFT JOIN news_stock_relation r ON n.id = r.news_id
            WHERE r.news_id IS NULL
            """
            params = None
        
        # 執行查詢
        df = pd.read_sql(query, self.conn, params=params)
        
        if df.empty:
            self.logger.info("未找到需要分析的新聞")
            return 0
        
        # 載入股票資訊
        stock_info = pd.read_sql("SELECT stock_id, 公司簡稱 FROM company_info", self.conn)
        
        self.logger.info(f"開始分析 {len(df)} 篇新聞的相關股票")
        
        # 相關股票分析結果
        results = []
        
        # 逐篇分析
        for _, row in df.iterrows():
            news_id = row['id']
            title = row['title']
            content = row['content'] if pd.notna(row['content']) else ""
            
            # 合併標題和內容，標題權重更高
            text = title + " " + title + " " + content
            
            # 查找股票相關性
            mentioned_stocks = {}
            
            # 查找股票代碼 (4或5位數字)
            stock_codes = re.findall(r'\b(\d{4,5})\b', text)
            for code in stock_codes:
                if code in stock_info['stock_id'].values:
                    mentioned_stocks[code] = mentioned_stocks.get(code, 0) + 1
            
            # 查找公司名稱
            for _, stock_row in stock_info.iterrows():
                stock_id = stock_row['stock_id']
                company_name = stock_row['公司簡稱']
                
                if pd.notna(company_name) and company_name in text:
                    mentioned_stocks[stock_id] = mentioned_stocks.get(stock_id, 0) + 2  # 公司名稱提及權重更高
            
            # 添加到結果列表
            for stock_id, count in mentioned_stocks.items():
                # 計算關聯度 (簡單實作)
                relevance = min(1.0, count / 10)
                
                results.append({
                    'news_id': news_id,
                    'stock_id': stock_id,
                    'relevance': relevance
                })
        
        # 批量保存分析結果
        cursor = self.conn.cursor()
        for result in results:
            cursor.execute("""
            INSERT OR REPLACE INTO news_stock_relation 
            (news_id, stock_id, relevance)
            VALUES (?, ?, ?)
            """, (
                result['news_id'],
                result['stock_id'],
                result['relevance']
            ))
        
        self.conn.commit()
        self.logger.info(f"成功分析 {len(df)} 篇新聞的相關股票，共找到 {len(results)} 個關聯")
        
        return len(df)
    
    def get_stock_news_sentiment(self, stock_id: str, days: int = 30) -> pd.DataFrame:
        """獲取特定股票的新聞情緒資料
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近幾天
        
        Returns:
            DataFrame: 新聞情緒資料
        """
        query = """
        SELECT n.id, n.title, n.publish_date, n.source, 
               s.sentiment_score, s.positive_score, s.negative_score, s.neutral_score, s.keywords,
               r.relevance
        FROM stock_news n
        JOIN news_stock_relation r ON n.id = r.news_id
        JOIN news_sentiment s ON n.id = s.news_id
        WHERE r.stock_id = ?
        AND n.publish_date >= date('now', '-{} days')
        ORDER BY n.publish_date DESC, r.relevance DESC
        """.format(days)
        
        df = pd.read_sql(query, self.conn, params=(stock_id,))
        
        return df
    
    def plot_sentiment_trend(self, stock_id: str, days: int = 30) -> plt.Figure:
        """繪製特定股票的新聞情緒趨勢圖
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近幾天
        
        Returns:
            Figure: 情緒趨勢圖
        """
        # 獲取新聞情緒資料
        df = self.get_stock_news_sentiment(stock_id, days)
        
        if df.empty:
            # 創建空圖表
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"沒有找到 {stock_id} 的相關新聞", ha='center', va='center', fontsize=14)
            return fig
        
        # 轉換日期格式
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        
        # 按日期分組，計算每日加權平均情緒分數
        df['weighted_score'] = df['sentiment_score'] * df['relevance']
        daily_sentiment = df.groupby('publish_date')['weighted_score'].mean().reset_index()
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 繪製情緒趨勢線
        ax.plot(daily_sentiment['publish_date'], daily_sentiment['weighted_score'], 'b-', linewidth=2, label='情緒得分')
        
        # 標記正面/負面區域
        ax.fill_between(daily_sentiment['publish_date'], daily_sentiment['weighted_score'], 0, 
                      where=(daily_sentiment['weighted_score'] >= 0), 
                      color='green', alpha=0.3, interpolate=True)
        ax.fill_between(daily_sentiment['publish_date'], daily_sentiment['weighted_score'], 0, 
                      where=(daily_sentiment['weighted_score'] <= 0), 
                      color='red', alpha=0.3, interpolate=True)
        
        # 添加0線
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # 設置標題和標籤
        ax.set_title(f"{stock_id} 最近 {days} 天新聞情緒趨勢", fontsize=16)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('情緒得分', fontsize=12)
        ax.set_ylim(-1, 1)
        
        # 設置網格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 美化圖表
        plt.tight_layout()
        
        return fig
    
    def get_news_keywords(self, stock_id: Optional[str] = None, days: int = 30, top_n: int = 20) -> pd.DataFrame:
        """獲取新聞關鍵字統計
        
        Args:
            stock_id (str, optional): 股票代碼，如果為None則分析所有新聞
            days (int): 最近幾天
            top_n (int): 返回前N個關鍵字
        
        Returns:
            DataFrame: 關鍵字統計
        """
        # 建立查詢
        if stock_id:
            query = """
            SELECT s.keywords
            FROM stock_news n
            JOIN news_sentiment s ON n.id = s.news_id
            JOIN news_stock_relation r ON n.id = r.news_id
            WHERE r.stock_id = ?
            AND n.publish_date >= date('now', '-{} days')
            """.format(days)
            params = (stock_id,)
        else:
            query = """
            SELECT s.keywords
            FROM stock_news n
            JOIN news_sentiment s ON n.id = s.news_id
            WHERE n.publish_date >= date('now', '-{} days')
            """.format(days)
            params = None
        
        # 執行查詢
        df = pd.read_sql(query, self.conn, params=params)
        
        if df.empty:
            return pd.DataFrame(columns=['keyword', 'count'])
        
        # 合併所有關鍵字並統計
        all_keywords = []
        for keywords_str in df['keywords']:
            if pd.notna(keywords_str) and keywords_str:
                all_keywords.extend(keywords_str.split(','))
        
        # 統計關鍵字頻率
        keyword_counts = Counter(all_keywords)
        
        # 轉換為DataFrame
        result_df = pd.DataFrame({
            'keyword': list(keyword_counts.keys()),
            'count': list(keyword_counts.values())
        })
        
        # 排序並取前N個
        result_df = result_df.sort_values('count', ascending=False).head(top_n)
        
        return result_df
    
    def plot_keyword_cloud(self, stock_id: Optional[str] = None, days: int = 30) -> plt.Figure:
        """繪製關鍵字雲圖
        
        Args:
            stock_id (str, optional): 股票代碼，如果為None則分析所有新聞
            days (int): 最近幾天
        
        Returns:
            Figure: 關鍵字雲圖
        """
        # 獲取關鍵字統計
        keyword_df = self.get_news_keywords(stock_id, days, top_n=50)
        
        if keyword_df.empty:
            # 創建空圖表
            fig, ax = plt.subplots(figsize=(10, 6))
            if stock_id:
                ax.text(0.5, 0.5, f"沒有找到 {stock_id} 的相關新聞關鍵字", ha='center', va='center', fontsize=14)
            else:
                ax.text(0.5, 0.5, "沒有找到任何新聞關鍵字", ha='center', va='center', fontsize=14)
            return fig
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 繪製水平條形圖
        bars = ax.barh(keyword_df['keyword'][:20], keyword_df['count'][:20], color='skyblue')
        
        # 添加數據標籤
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                   f"{width}", va='center')
        
        # 設置標題和標籤
        if stock_id:
            ax.set_title(f"{stock_id} 最近 {days} 天新聞關鍵字頻率", fontsize=16)
        else:
            ax.set_title(f"最近 {days} 天所有新聞關鍵字頻率", fontsize=16)
        
        ax.set_xlabel('出現次數', fontsize=12)
        ax.set_ylabel('關鍵字', fontsize=12)
        
        # 翻轉Y軸，使最高頻的關鍵字在上方
        ax.invert_yaxis()
        
        # 設置網格
        ax.grid(True, linestyle='--', alpha=0.6, axis='x')
        
        # 美化圖表
        plt.tight_layout()
        
        return fig
    
    def analyze_news_topics(self, days: int = 30, top_n: int = 5) -> List[Dict]:
        """分析熱門新聞主題
        
        Args:
            days (int): 最近幾天
            top_n (int): 返回前N個主題
        
        Returns:
            list: 熱門主題列表
        """
        # 查詢最近的新聞
        query = """
        SELECT n.id, n.title, n.content, n.publish_date
        FROM stock_news n
        WHERE n.publish_date >= date('now', '-{} days')
        """.format(days)
        
        df = pd.read_sql(query, self.conn)
        
        if df.empty:
            return []
        
        # 合併標題和內容
        df['text'] = df['title'] + " " + df['content'].fillna("")
        
        # 提取關鍵詞
        topic_keywords = []
        for text in df['text']:
            keywords = jieba.analyse.extract_tags(text, topK=5)
            topic_keywords.extend(keywords)
        
        # 統計關鍵詞頻率
        keyword_counts = Counter(topic_keywords)
        
        # 獲取前N個關鍵詞
        top_keywords = keyword_counts.most_common(top_n)
        
        # 為每個關鍵詞找出相關的新聞
        topics = []
        for keyword, count in top_keywords:
            related_news = []
            for _, row in df.iterrows():
                if keyword in row['text']:
                    related_news.append({
                        'id': row['id'],
                        'title': row['title'],
                        'date': row['publish_date']
                    })
            
            topics.append({
                'keyword': keyword,
                'count': count,
                'related_news': related_news[:5]  # 只取前5篇相關新聞
            })
        
        return topics
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()