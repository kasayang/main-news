# data_collectors/news_collector.py
import requests
from bs4 import BeautifulSoup
import logging
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import re

class NewsCollector:
    """新聞資料收集器，負責從各種來源收集與股票相關的新聞"""
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """初始化新聞收集器
        
        Args:
            database_path (str): 資料庫路徑
        """
        self.database_path = database_path
        self.conn = sqlite3.connect(database_path)
        
        # 設置日誌
        self.logger = logging.getLogger("NewsCollector")
        
        # 建立資料表
        self._create_tables()
    
    def _create_tables(self):
        """建立新聞相關資料表"""
        cursor = self.conn.cursor()
        
        # 新聞資料表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            url TEXT,
            publish_date DATE,
            source TEXT,
            fetch_date DATE,
            UNIQUE(url, publish_date)
        )
        ''')
        
        # 新聞-股票關聯表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_stock_relation (
            news_id INTEGER,
            stock_id TEXT,
            relevance REAL,  -- 關聯度評分 (0-1)
            FOREIGN KEY (news_id) REFERENCES stock_news(id),
            PRIMARY KEY (news_id, stock_id)
        )
        ''')
        
        # 新聞情緒分析表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_sentiment (
            news_id INTEGER PRIMARY KEY,
            sentiment_score REAL,  -- 情緒評分 (-1 負面, 0 中性, 1 正面)
            positive_score REAL,   -- 正面評分 (0-1)
            negative_score REAL,   -- 負面評分 (0-1)
            neutral_score REAL,    -- 中性評分 (0-1)
            keywords TEXT,         -- 關鍵字，以逗號分隔
            summary TEXT,          -- 摘要
            FOREIGN KEY (news_id) REFERENCES stock_news(id)
        )
        ''')
        
        self.conn.commit()
    
    def collect_twse_news(self, days: int = 7) -> List[Dict]:
        """從證交所收集最近的新聞
        
        Args:
            days (int): 收集最近幾天的新聞
        
        Returns:
            list: 新聞資料列表
        """
        base_url = "https://www.twse.com.tw/zh/news/newslist.html"
        news_list = []
        
        try:
            self.logger.info("開始從證交所收集新聞...")
            response = requests.get(base_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_elements = soup.find_all('div', class_='news-item')
            
            for element in news_elements:
                try:
                    title = element.find('h3').text.strip()
                    date_text = element.find('span', class_='date').text.strip()
                    link = element.find('a')['href']
                    
                    # 解析日期
                    publish_date = self._parse_date(date_text)
                    
                    # 獲取詳細內容
                    content = self._fetch_news_content(link)
                    
                    # 保存新聞
                    news_data = {
                        'title': title,
                        'content': content,
                        'url': link,
                        'publish_date': publish_date,
                        'source': '證交所',
                        'fetch_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    # 保存到資料庫
                    news_id = self._save_news_to_db(news_data)
                    
                    if news_id:
                        # 分析新聞中提到的股票
                        self._analyze_stock_mentions(news_id, title + " " + content)
                        
                        # 進行情緒分析
                        self._analyze_sentiment(news_id, title, content)
                        
                        news_list.append(news_data)
                
                except Exception as item_error:
                    self.logger.warning(f"處理單一新聞項目時發生錯誤: {item_error}")
            
            self.logger.info(f"成功從證交所收集 {len(news_list)} 則新聞")
            return news_list
        
        except Exception as e:
            self.logger.error(f"從證交所收集新聞時發生錯誤: {e}")
            return []
    
    def collect_money_news(self, days: int = 7) -> List[Dict]:
        """從經濟日報收集最近的新聞
        
        Args:
            days (int): 收集最近幾天的新聞
        
        Returns:
            list: 新聞資料列表
        """
        base_url = "https://money.udn.com/money/index"
        news_list = []
        
        try:
            self.logger.info("開始從經濟日報收集新聞...")
            response = requests.get(base_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 根據實際網站結構調整選擇器
            news_elements = soup.find_all('div', class_='story__content')
            
            for element in news_elements:
                try:
                    title = element.find('h3').text.strip()
                    link = element.find('a')['href']
                    
                    # 獲取詳細內容
                    content = self._fetch_news_content(link)
                    
                    # 解析日期（可能需要調整）
                    publish_date = datetime.now().strftime('%Y-%m-%d')
                    
                    # 保存新聞
                    news_data = {
                        'title': title,
                        'content': content,
                        'url': link,
                        'publish_date': publish_date,
                        'source': '經濟日報',
                        'fetch_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    # 保存到資料庫
                    news_id = self._save_news_to_db(news_data)
                    
                    if news_id:
                        # 分析新聞中提到的股票
                        self._analyze_stock_mentions(news_id, title + " " + content)
                        
                        # 進行情緒分析
                        self._analyze_sentiment(news_id, title, content)
                        
                        news_list.append(news_data)
                
                except Exception as item_error:
                    self.logger.warning(f"處理單一新聞項目時發生錯誤: {item_error}")
            
            self.logger.info(f"成功從經濟日報收集 {len(news_list)} 則新聞")
            return news_list
        
        except Exception as e:
            self.logger.error(f"從經濟日報收集新聞時發生錯誤: {e}")
            return []

    def collect_integrated_news(self, days: int = 7) -> List[Dict]:
        """整合收集多個來源的新聞
        
        Args:
            days (int): 收集最近幾天的新聞
        
        Returns:
            list: 新聞資料列表
        """
        # 收集不同來源的新聞
        twse_news = self.collect_twse_news(days)
        money_news = self.collect_money_news(days)
        
        # 合併新聞列表
        all_news = twse_news + money_news
        
        # 去重（根據 URL）
        unique_news = []
        seen_urls = set()
        
        for news in all_news:
            if news['url'] not in seen_urls:
                unique_news.append(news)
                seen_urls.add(news['url'])
        
        self.logger.info(f"總共收集 {len(unique_news)} 則唯一新聞")
        
        return unique_news
    
    def _fetch_news_content(self, url: str) -> str:
        """抓取新聞內容
        
        Args:
            url (str): 新聞URL
        
        Returns:
            str: 新聞內容
        """
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 嘗試找到新聞內容 (這裡的選擇器需要根據實際網站結構調整)
            content_div = soup.find('div', class_='content')
            
            if content_div:
                return content_div.get_text(strip=True)
            else:
                return ""
        except Exception as e:
            self.logger.warning(f"抓取新聞內容時發生錯誤: {e}")
            return ""
    
    def _save_news_to_db(self, news_data: Dict) -> Optional[int]:
        """將新聞保存到資料庫
        
        Args:
            news_data (dict): 新聞資料
        
        Returns:
            int: 新聞ID
        """
        try:
            cursor = self.conn.cursor()
            
            # 檢查新聞是否已存在
            cursor.execute(
                "SELECT id FROM stock_news WHERE url = ? AND publish_date = ?",
                (news_data['url'], news_data['publish_date'])
            )
            
            result = cursor.fetchone()
            
            if result:
                # 新聞已存在，返回其ID
                return result[0]
            
            # 插入新聞
            cursor.execute('''
            INSERT INTO stock_news 
            (title, content, url, publish_date, source, fetch_date)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                news_data['title'],
                news_data['content'],
                news_data['url'],
                news_data['publish_date'],
                news_data['source'],
                news_data['fetch_date']
            ))
            
            self.conn.commit()
            
            # 獲取插入的ID
            return cursor.lastrowid
        
        except Exception as e:
            self.logger.error(f"保存新聞到資料庫時發生錯誤: {e}")
            return None
    
    def _analyze_stock_mentions(self, news_id: int, text: str):
        """分析新聞中提到的股票
        
        Args:
            news_id (int): 新聞ID
            text (str): 要分析的文本
        """
        try:
            # 載入股票資訊
            stock_info = pd.read_sql("SELECT stock_id, 公司簡稱 FROM company_info", self.conn)
            
            mentioned_stocks = {}
            
            # 查找股票代碼 (4或5位數字)
            stock_codes = re.findall(r'\b(\d{4,5})\b', text)
            for code in stock_codes:
                if code in stock_info['stock_id'].values:
                    mentioned_stocks[code] = mentioned_stocks.get(code, 0) + 1
            
            # 查找公司名稱
            for _, row in stock_info.iterrows():
                stock_id = row['stock_id']
                company_name = row['公司簡稱']
                
                if pd.notna(company_name) and company_name in text:
                    mentioned_stocks[stock_id] = mentioned_stocks.get(stock_id, 0) + 2  # 公司名稱提及權重更高
            
            # 保存結果
            cursor = self.conn.cursor()
            
            for stock_id, count in mentioned_stocks.items():
                # 計算關聯度 (簡單實作)
                relevance = min(1.0, count / 10)
                
                cursor.execute('''
                INSERT OR REPLACE INTO news_stock_relation 
                (news_id, stock_id, relevance)
                VALUES (?, ?, ?)
                ''', (news_id, stock_id, relevance))
            
            self.conn.commit()
        
        except Exception as e:
            self.logger.error(f"分析新聞中提到的股票時發生錯誤: {e}")
    
    def _analyze_sentiment(self, news_id: int, title: str, content: str):
        """進行新聞情緒分析
        
        Args:
            news_id (int): 新聞ID
            title (str): 新聞標題
            content (str): 新聞內容
        """
        try:
            # 這裡可以使用NLP庫進行情緒分析，以下是示範
            # 簡單實作：尋找正面和負面關鍵詞
            
            positive_words = ["上漲", "成長", "獲利", "增加", "好", "看好", "強", "正面", "樂觀"]
            negative_words = ["下跌", "虧損", "衰退", "減少", "差", "看空", "弱", "負面", "悲觀"]
            
            text = title + " " + content
            
            # 統計關鍵詞出現次數
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            total_count = positive_count + negative_count
            
            # 計算得分
            if total_count > 0:
                positive_score = positive_count / total_count
                negative_score = negative_count / total_count
                sentiment_score = positive_score - negative_score
            else:
                positive_score = 0.5
                negative_score = 0.5
                sentiment_score = 0
            
            neutral_score = 1 - abs(sentiment_score)
            
            # 提取關鍵字 (簡易實作)
            keywords = []
            for word in positive_words:
                if word in text:
                    keywords.append(word)
            for word in negative_words:
                if word in text:
                    keywords.append(word)
            
            # 生成摘要 (簡易實作)
            summary = title
            
            # 保存結果
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO news_sentiment 
            (news_id, sentiment_score, positive_score, negative_score, neutral_score, keywords, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                news_id, 
                sentiment_score, 
                positive_score, 
                negative_score, 
                neutral_score,
                ",".join(keywords),
                summary
            ))
            
            self.conn.commit()
        
        except Exception as e:
            self.logger.error(f"進行新聞情緒分析時發生錯誤: {e}")
    
    def _parse_date(self, date_text: str) -> str:
        """解析日期字符串
        
        Args:
            date_text (str): 日期字符串
        
        Returns:
            str: 格式化日期 (YYYY-MM-DD)
        """
        try:
            # 處理不同的日期格式
            patterns = [
                r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',  # YYYY-MM-DD 或 YYYY/MM/DD
                r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',  # DD-MM-YYYY 或 DD/MM/YYYY
                r'(\d{4})年(\d{1,2})月(\d{1,2})日'     # YYYY年MM月DD日
            ]
            
            for pattern in patterns:
                match = re.search(pattern, date_text)
                if match:
                    groups = match.groups()
                    if len(groups[0]) == 4:  # 第一組是年份
                        year, month, day = groups
                    else:  # 第三組是年份
                        day, month, year = groups
                    
                    return f"{year}-{int(month):02d}-{int(day):02d}"
            
            # 若無法解析，返回當天日期
            return datetime.now().strftime('%Y-%m-%d')
        
        except Exception as e:
            self.logger.warning(f"解析日期時發生錯誤: {e}")
            return datetime.now().strftime('%Y-%m-%d')
    
    def get_news_for_stock(self, stock_id: str, days: int = 30) -> pd.DataFrame:
        """獲取特定股票的相關新聞
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近幾天的新聞
        
        Returns:
            DataFrame: 新聞資料
        """
        query = f"""
        SELECT n.id, n.title, n.content, n.url, n.publish_date, n.source,
               r.relevance, s.sentiment_score, s.positive_score, s.negative_score
        FROM stock_news n
        JOIN news_stock_relation r ON n.id = r.news_id
        LEFT JOIN news_sentiment s ON n.id = s.news_id
        WHERE r.stock_id = ?
        AND n.publish_date >= date('now', '-{days} days')
        ORDER BY n.publish_date DESC, r.relevance DESC
        """
        
        df = pd.read_sql(query, self.conn, params=(stock_id,))
        return df
    
    def get_sentiment_summary(self, stock_id: str, days: int = 30) -> Dict:
        """獲取特定股票的新聞情緒摘要
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近幾天的新聞
        
        Returns:
            dict: 情緒摘要
        """
        news_df = self.get_news_for_stock(stock_id, days)
        
        if news_df.empty:
            return {
                'stock_id': stock_id,
                'news_count': 0,
                'avg_sentiment': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0
            }
        
        # 計算平均情緒分數
        avg_sentiment = news_df['sentiment_score'].mean()
        
        # 計算正面、負面、中性新聞比例
        positive_news = news_df[news_df['sentiment_score'] > 0.2]
        negative_news = news_df[news_df['sentiment_score'] < -0.2]
        neutral_news = news_df[(news_df['sentiment_score'] >= -0.2) & (news_df['sentiment_score'] <= 0.2)]
        
        total_news = len(news_df)
        
        return {
            'stock_id': stock_id,
            'news_count': total_news,
            'avg_sentiment': avg_sentiment,
            'positive_ratio': len(positive_news) / total_news if total_news > 0 else 0,
            'negative_ratio': len(negative_news) / total_news if total_news > 0 else 0,
            'neutral_ratio': len(neutral_news) / total_news if total_news > 0 else 0
        }
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()