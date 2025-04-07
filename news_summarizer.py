# news_summarizer.py
import sqlite3
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

class NewsSummarizer:
    """新聞摘要自動生成模組
    
    此模組使用自然語言處理技術從多篇新聞中提取關鍵信息，並生成摘要。
    支援的摘要方法：
    - 基於 TF-IDF 的提取式摘要
    - 基於主題模型的摘要
    - 時間序列摘要（追蹤事件發展）
    """
    
    def __init__(self, database_path: str = "tw_stock_data.db"):
        """初始化新聞摘要器
        
        Args:
            database_path (str): 資料庫路徑
        """
        self.database_path = database_path
        self.conn = sqlite3.connect(database_path)
        
        # 設置日誌
        self.logger = logging.getLogger("NewsSummarizer")
        
        # 初始化結巴斷詞
        try:
            jieba.load_userdict("dict/financial_terms.txt")
        except:
            self.logger.info("未找到自定義財經詞典，使用預設詞典")
    
    def get_stock_news(self, stock_id: Optional[str] = None, days: int = 30, limit: int = 100) -> pd.DataFrame:
        """獲取股票相關新聞
        
        Args:
            stock_id (str, optional): 股票代碼，如果為 None 則獲取所有新聞
            days (int): 最近幾天的新聞
            limit (int): 返回的最大新聞數量
        
        Returns:
            DataFrame: 新聞資料
        """
        # 計算日期範圍
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        if stock_id:
            # 查詢特定股票相關新聞
            query = """
            SELECT n.id, n.title, n.content, n.publish_date, n.source,
                   s.sentiment_score, r.relevance
            FROM stock_news n
            JOIN news_stock_relation r ON n.id = r.news_id
            LEFT JOIN news_sentiment s ON n.id = s.news_id
            WHERE r.stock_id = ?
            AND n.publish_date BETWEEN ? AND ?
            ORDER BY r.relevance DESC
            LIMIT ?
            """
            params = (stock_id, start_date, end_date, limit)
        else:
            # 查詢所有新聞
            query = """
            SELECT n.id, n.title, n.content, n.publish_date, n.source,
                   s.sentiment_score
            FROM stock_news n
            LEFT JOIN news_sentiment s ON n.id = s.news_id
            WHERE n.publish_date BETWEEN ? AND ?
            ORDER BY n.publish_date DESC
            LIMIT ?
            """
            params = (start_date, end_date, limit)
        
        df = pd.read_sql(query, self.conn, params=params)
        
        # 確保內容欄位不是空值
        df['content'] = df['content'].fillna('')
        
        return df
    
    def get_news_by_topic(self, keywords: List[str], days: int = 30, limit: int = 100) -> pd.DataFrame:
        """獲取特定主題的新聞
        
        Args:
            keywords (List[str]): 關鍵詞列表
            days (int): 最近幾天的新聞
            limit (int): 返回的最大新聞數量
        
        Returns:
            DataFrame: 新聞資料
        """
        # 計算日期範圍
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # 建立SQL條件
        keyword_conditions = []
        params = [start_date, end_date]
        
        for keyword in keywords:
            keyword_conditions.append("(n.title LIKE ? OR n.content LIKE ?)")
            params.extend([f'%{keyword}%', f'%{keyword}%'])
        
        keyword_sql = " OR ".join(keyword_conditions)
        
        # 查詢特定主題相關新聞
        query = f"""
        SELECT n.id, n.title, n.content, n.publish_date, n.source,
               s.sentiment_score
        FROM stock_news n
        LEFT JOIN news_sentiment s ON n.id = s.news_id
        WHERE ({keyword_sql})
        AND n.publish_date BETWEEN ? AND ?
        ORDER BY n.publish_date DESC
        LIMIT ?
        """
        
        params.extend([limit])
        
        df = pd.read_sql(query, self.conn, params=params)
        
        # 確保內容欄位不是空值
        df['content'] = df['content'].fillna('')
        
        return df
    
    def extract_key_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """從文本中提取關鍵句子
        
        使用 TF-IDF 算法評分每個句子的重要性
        
        Args:
            text (str): 要提取的文本
            num_sentences (int): 提取的句子數量
        
        Returns:
            List[str]: 關鍵句子列表
        """
        # 分句
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # 對每個句子進行斷詞
        sentence_words = []
        for sentence in sentences:
            words = [word for word in jieba.cut(sentence) if len(word.strip()) > 1]
            sentence_words.append(' '.join(words))
        
        # 計算 TF-IDF
        vectorizer = TfidfVectorizer()
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentence_words)
            
            # 計算每個句子的重要性分數
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = np.sum(tfidf_matrix[i].toarray())
                sentence_scores.append((i, score, sentence))
            
            # 按分數排序
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 按原始順序排序前N個重要句子
            top_sentences = [(idx, sentence) for idx, _, sentence in sentence_scores[:num_sentences]]
            top_sentences.sort(key=lambda x: x[0])
            
            return [sentence for _, sentence in top_sentences]
        
        except:
            # 如果向量化失敗，返回前 num_sentences 個句子
            return sentences[:num_sentences]
    
    def extract_key_phrases(self, text: str, num_phrases: int = 10) -> List[str]:
        """從文本中提取關鍵短語
        
        Args:
            text (str): 要提取的文本
            num_phrases (int): 提取的短語數量
        
        Returns:
            List[str]: 關鍵短語列表
        """
        try:
            # 使用 jieba 的 TextRank 算法提取關鍵詞
            key_phrases = jieba.analyse.textrank(text, topK=num_phrases, withWeight=False)
            return key_phrases
        except:
            # 如果提取失敗，使用 TF-IDF 算法
            try:
                key_phrases = jieba.analyse.extract_tags(text, topK=num_phrases, withWeight=False)
                return key_phrases
            except:
                return []
    
    def discover_topics(self, news_df: pd.DataFrame, num_topics: int = 3, num_words: int = 8) -> List[Dict]:
        """發現新聞中的主題
        
        使用 LDA 主題模型
        
        Args:
            news_df (DataFrame): 新聞資料框
            num_topics (int): 主題數量
            num_words (int): 每個主題包含的詞彙數量
        
        Returns:
            List[Dict]: 主題列表，每個主題包含詞彙和相關新聞
        """
        if news_df.empty:
            return []
        
        # 合併標題和內容
        texts = news_df['title'] + ' ' + news_df['content']
        
        # 對每篇新聞進行斷詞
        processed_texts = []
        for text in texts:
            words = [word for word in jieba.cut(text) if len(word.strip()) > 1]
            processed_texts.append(' '.join(words))
        
        # 計算詞頻
        vectorizer = CountVectorizer(max_features=1000)
        
        try:
            count_matrix = vectorizer.fit_transform(processed_texts)
            
            # LDA 主題模型
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                max_iter=10,
                random_state=0
            )
            
            # 訓練 LDA 模型
            lda.fit(count_matrix)
            
            # 獲取每個主題的詞彙
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                # 獲取前 num_words 個詞
                top_words_idx = topic.argsort()[:-num_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                # 找出最相關的新聞
                topic_scores = lda.transform(count_matrix)[:, topic_idx]
                top_news_idx = topic_scores.argsort()[-5:][::-1]
                related_news = news_df.iloc[top_news_idx][['id', 'title', 'publish_date', 'source']].to_dict('records')
                
                topics.append({
                    'id': topic_idx,
                    'keywords': top_words,
                    'related_news': related_news
                })
            
            return topics
        
        except Exception as e:
            self.logger.error(f"主題發現失敗: {str(e)}")
            return []
    
    def generate_extractive_summary(self, news_df: pd.DataFrame, max_sentences: int = 5) -> Dict:
        """生成提取式摘要
        
        從多篇新聞中提取關鍵句子生成摘要
        
        Args:
            news_df (DataFrame): 新聞資料
            max_sentences (int): 最大句子數量
        
        Returns:
            Dict: 摘要資訊
        """
        if news_df.empty:
            return {
                'summary': '',
                'key_phrases': [],
                'source_count': 0,
                'date_range': ''
            }
        
        # 對新聞按情緒分數排序，並優先使用高關聯度的新聞
        if 'relevance' in news_df.columns:
            sorted_news = news_df.sort_values(['relevance', 'publish_date'], ascending=[False, False])
        else:
            sorted_news = news_df.sort_values('publish_date', ascending=False)
        
        # 獲取最近的新聞
        recent_news = sorted_news.head(10)
        
        # 合併標題
        combined_title = ' '.join(recent_news['title'])
        
        # 提取所有新聞中的關鍵詞
        all_text = combined_title + ' ' + ' '.join(recent_news['content'])
        key_phrases = self.extract_key_phrases(all_text, 15)
        
        # 從每篇新聞中提取關鍵句子
        all_sentences = []
        
        for _, news in recent_news.iterrows():
            text = f"{news['title']}。{news['content']}"
            sentences = self.extract_key_sentences(text, 2)
            
            if sentences:
                source_prefix = f"【{news['source']}】" if 'source' in news else ""
                for sentence in sentences:
                    # 確保句子足夠長（避免無意義的短句）
                    if len(sentence) > 10:
                        all_sentences.append(f"{source_prefix}{sentence}。")
        
        # 限制句子總數
        all_sentences = all_sentences[:max_sentences]
        
        # 生成摘要
        summary = ''.join(all_sentences)
        
        # 計算日期範圍
        if 'publish_date' in news_df.columns:
            min_date = pd.to_datetime(news_df['publish_date']).min()
            max_date = pd.to_datetime(news_df['publish_date']).max()
            
            if min_date == max_date:
                date_range = min_date.strftime('%Y-%m-%d')
            else:
                date_range = f"{min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')}"
        else:
            date_range = ''
        
        return {
            'summary': summary,
            'key_phrases': key_phrases,
            'source_count': len(news_df['source'].unique()) if 'source' in news_df.columns else 0,
            'date_range': date_range
        }
    
    def generate_topic_summary(self, stock_id: Optional[str] = None, days: int = 7) -> Dict:
        """生成主題摘要
        
        分析最近新聞中的主題，並為每個主題生成摘要
        
        Args:
            stock_id (str, optional): 股票代碼，如果為 None 則分析所有新聞
            days (int): 最近幾天的新聞
        
        Returns:
            Dict: 主題摘要
        """
        # 獲取相關新聞
        news_df = self.get_stock_news(stock_id, days, 100)
        
        if news_df.empty:
            return {
                'status': 'error',
                'message': '沒有找到相關新聞'
            }
        
        # 發現主題
        topics = self.discover_topics(news_df)
        
        # 為每個主題生成摘要
        for topic in topics:
            # 獲取主題相關的新聞
            topic_news_ids = [news['id'] for news in topic['related_news']]
            topic_news = news_df[news_df['id'].isin(topic_news_ids)]
            
            # 生成摘要
            summary_result = self.generate_extractive_summary(topic_news, 3)
            topic['summary'] = summary_result['summary']
        
        # 整體摘要
        general_summary = self.generate_extractive_summary(news_df)
        
        # 準備結果
        result = {
            'status': 'success',
            'stock_id': stock_id,
            'days': days,
            'general_summary': general_summary,
            'topics': topics,
            'news_count': len(news_df),
            'date_range': general_summary['date_range']
        }
        
        return result
    
    def generate_timeline_summary(self, stock_id: str, days: int = 30) -> Dict:
        """生成時間線摘要
        
        按時間順序追蹤事件發展
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近幾天的新聞
        
        Returns:
            Dict: 時間線摘要
        """
        # 獲取相關新聞
        news_df = self.get_stock_news(stock_id, days, 100)
        
        if news_df.empty:
            return {
                'status': 'error',
                'message': '沒有找到相關新聞'
            }
        
        # 按發布日期分組
        news_df['publish_date'] = pd.to_datetime(news_df['publish_date'])
        grouped_news = news_df.groupby(news_df['publish_date'].dt.date)
        
        # 為每一天生成摘要
        timeline = []
        
        for date, group in grouped_news:
            # 按情緒分數排序
            if 'sentiment_score' in group.columns:
                # 先處理正面新聞，再處理負面新聞，然後是中性新聞
                sorted_group = group.sort_values('sentiment_score', ascending=False)
            else:
                sorted_group = group
            
            # 生成每日摘要
            daily_summary = self.generate_extractive_summary(sorted_group, 2)
            
            # 計算平均情緒
            avg_sentiment = group['sentiment_score'].mean() if 'sentiment_score' in group.columns else None
            
            # 添加到時間線
            timeline.append({
                'date': date.strftime('%Y-%m-%d'),
                'news_count': len(group),
                'summary': daily_summary['summary'],
                'key_phrases': daily_summary['key_phrases'][:5],
                'avg_sentiment': float(avg_sentiment) if pd.notna(avg_sentiment) else None,
                'sources': group['source'].unique().tolist() if 'source' in group.columns else []
            })
        
        # 按日期排序時間線
        timeline.sort(key=lambda x: x['date'])
        
        # 提取整體關鍵詞和摘要
        general_summary = self.generate_extractive_summary(news_df)
        
        # 準備結果
        result = {
            'status': 'success',
            'stock_id': stock_id,
            'days': days,
            'news_count': len(news_df),
            'date_range': general_summary['date_range'],
            'key_phrases': general_summary['key_phrases'],
            'timeline': timeline
        }
        
        return result
    
    def generate_stock_news_report(self, stock_id: str, days: int = 7) -> Dict:
        """生成股票新聞報告
        
        綜合多種摘要方法，生成完整的新聞分析報告
        
        Args:
            stock_id (str): 股票代碼
            days (int): 最近幾天的新聞
        
        Returns:
            Dict: 新聞報告
        """
        # 獲取相關新聞
        news_df = self.get_stock_news(stock_id, days, 200)
        
        if news_df.empty:
            return {
                'status': 'error',
                'message': '沒有找到相關新聞'
            }
        
        # 生成整體摘要
        general_summary = self.generate_extractive_summary(news_df)
        
        # 發現主題
        topics = self.discover_topics(news_df)
        
        # 生成時間線摘要
        news_df['publish_date'] = pd.to_datetime(news_df['publish_date'])
        latest_date = news_df['publish_date'].max().date()
        
        # 按情緒分類新聞
        positive_news = news_df[news_df['sentiment_score'] > 0.2] if 'sentiment_score' in news_df.columns else pd.DataFrame()
        negative_news = news_df[news_df['sentiment_score'] < -0.2] if 'sentiment_score' in news_df.columns else pd.DataFrame()
        
        positive_summary = self.generate_extractive_summary(positive_news, 3) if not positive_news.empty else {'summary': '', 'key_phrases': []}
        negative_summary = self.generate_extractive_summary(negative_news, 3) if not negative_news.empty else {'summary': '', 'key_phrases': []}
        
        # 計算情緒分佈
        sentiment_distribution = {}
        
        if 'sentiment_score' in news_df.columns:
            positive_ratio = len(positive_news) / len(news_df) if len(news_df) > 0 else 0
            negative_ratio = len(negative_news) / len(news_df) if len(news_df) > 0 else 0
            neutral_ratio = 1 - positive_ratio - negative_ratio
            
            sentiment_distribution = {
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': neutral_ratio,
                'avg_sentiment': float(news_df['sentiment_score'].mean()) if not news_df.empty else 0
            }
        
        # 最重要的新聞
        if 'relevance' in news_df.columns:
            important_news = news_df.sort_values('relevance', ascending=False).head(5)
        else:
            important_news = news_df.head(5)
        
        important_news_list = important_news[['id', 'title', 'publish_date', 'source']].to_dict('records')
        
        # 生成報告
        report = {
            'status': 'success',
            'stock_id': stock_id,
            'days': days,
            'news_count': len(news_df),
            'date_range': general_summary['date_range'],
            'general_summary': general_summary['summary'],
            'key_phrases': general_summary['key_phrases'],
            'sentiment_distribution': sentiment_distribution,
            'topics': topics,
            'positive_summary': positive_summary['summary'],
            'negative_summary': negative_summary['summary'],
            'important_news': important_news_list,
            'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else None
        }
        
        return report
    
    def generate_market_news_report(self, days: int = 3) -> Dict:
        """生成市場新聞報告
        
        分析整體市場的新聞
        
        Args:
            days (int): 最近幾天的新聞
        
        Returns:
            Dict: 市場新聞報告
        """
        # 獲取相關新聞 (無特定股票)
        news_df = self.get_stock_news(None, days, 300)
        
        if news_df.empty:
            return {
                'status': 'error',
                'message': '沒有找到相關新聞'
            }
        
        # 發現主題
        topics = self.discover_topics(news_df, num_topics=5)
        
        # 獲取熱門股票
        query = """
        SELECT r.stock_id, 
               c.公司簡稱 as company_name,
               COUNT(*) as news_count,
               AVG(s.sentiment_score) as avg_sentiment
        FROM news_stock_relation r
        JOIN stock_news n ON r.news_id = n.id
        LEFT JOIN news_sentiment s ON n.id = s.news_id
        LEFT JOIN company_info c ON r.stock_id = c.stock_id
        WHERE n.publish_date >= date('now', '-{} days')
        GROUP BY r.stock_id
        ORDER BY news_count DESC
        LIMIT 10
        """.format(days)
        
        hot_stocks_df = pd.read_sql(query, self.conn)
        
        hot_stocks = []
        for _, row in hot_stocks_df.iterrows():
            hot_stocks.append({
                'stock_id': row['stock_id'],
                'company_name': row['company_name'] if pd.notna(row['company_name']) else '',
                'news_count': int(row['news_count']),
                'avg_sentiment': float(row['avg_sentiment']) if pd.notna(row['avg_sentiment']) else 0
            })
        
        # 生成整體摘要
        general_summary = self.generate_extractive_summary(news_df)
        
        # 生成報告
        report = {
            'status': 'success',
            'days': days,
            'news_count': len(news_df),
            'date_range': general_summary['date_range'],
            'general_summary': general_summary['summary'],
            'key_phrases': general_summary['key_phrases'],
            'topics': topics,
            'hot_stocks': hot_stocks
        }
        
        return report
    
    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()