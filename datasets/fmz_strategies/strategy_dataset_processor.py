#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import markdownify
import re
import os
import time
import random
from pathlib import Path
import logging
from urllib.parse import urlparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fetch_myquant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MyQuantStrategyFetcher:
    def __init__(self, target_urls, output_dir):
        """初始化抓取器，支持多个URL"""
        self.target_urls = target_urls
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
        }
        self.session = requests.Session()
        self.session.headers = self.headers
    
    def fetch_webpage(self, url):
        """获取单个网页内容"""
        try:
            # 随机延时避免反爬（2-7秒）
            time.sleep(random.uniform(2, 7))
            logger.info(f"正在获取网页: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            response.encoding = 'utf-8'  # 确保中文正确解析
            logger.info(f"成功获取网页，状态码: {response.status_code}")
            return response.text
        except requests.RequestException as e:
            logger.error(f"获取网页 {url} 失败: {e}")
            return None
    
    def extract_strategies(self, html_content, url):
        """从HTML中提取策略内容，优化分割逻辑"""
        if not html_content:
            return []
        
        logger.info("开始解析网页内容...")
        soup = BeautifulSoup(html_content, 'html.parser')
        strategies = []
        
        # 提取网页标题作为文件名基础
        page_title = soup.title.text.strip() if soup.title else "myquant_strategy"
        page_title = re.sub(r'[\\/*?:"<>|]', '_', page_title)  # 清理非法字符
        
        # 提取主要内容区域（适配不同页面结构）
        main_content = soup.find('main') or soup.find('article')
        if not main_content:
            main_content = soup.find('div', class_=re.compile('content|article|strategy|post'))
        
        if not main_content:
            logger.warning(f"未找到主要内容区域，使用整个网页内容 (URL: {url})")
            main_content = soup
        
        # 转换为Markdown
        markdown_content = markdownify.markdownify(str(main_content), heading_style="atx")
        
        # 优化策略块分割：按二级标题（##）或一级标题（#）分割
        blocks = re.split(r'^##\s+|^#\s+', markdown_content, flags=re.MULTILINE)
        
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
                
            # 提取策略标题（取第一行非空行为标题）
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            title = lines[0] if lines else f"策略{i+1}"
            
            # 提取策略内容（标题之后的部分）
            content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            strategies.append({
                "title": title,
                "content": content,
                "url": url,
                "page_title": page_title
            })
        
        logger.info(f"从 {url} 解析出 {len(strategies)} 个策略块")
        return strategies
    
    def save_strategies(self, strategies, url):
        """保存策略到文件，按URL分组"""
        if not strategies:
            logger.warning(f"没有可保存的策略 (URL: {url})")
            return
        
        # 基于URL生成唯一目录
        url_path = urlparse(url).path
        url_name = re.sub(r'[\\/*?:"<>|]', '_', url_path)
        url_dir = self.output_dir / f"url_{url_name}"
        url_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整内容
        full_md_file = url_dir / f"{strategies[0]['page_title']}.md"
        with open(full_md_file, 'w', encoding='utf-8') as f:
            f.write(markdownify.markdownify(
                str(BeautifulSoup(self.fetch_webpage(url), 'html.parser').find('body')),
                heading_style="atx"
            ))
        logger.info(f"完整内容已保存至: {full_md_file}")
        
        # 保存每个策略块
        for i, strategy in enumerate(strategies):
            # 清理文件名
            filename = re.sub(r'[\\/*?:"<>|]', '_', strategy['title'])
            filename = f"{i+1}_{filename}.md" if filename else f"strategy_{i+1}.md"
            file_path = url_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {strategy['title']}\n\n")
                f.write(strategy['content'])
            
            logger.info(f"策略已保存至: {file_path}")
    
    def run(self):
        """执行多URL抓取流程"""
        logger.info(f"开始从 {len(self.target_urls)} 个URL提取策略...")
        for url in self.target_urls:
            html_content = self.fetch_webpage(url)
            if html_content:
                strategies = self.extract_strategies(html_content, url)
                self.save_strategies(strategies, url)
            else:
                logger.warning(f"跳过URL: {url}（内容为空）")
        logger.info("所有URL策略提取完成")

def main():
    # 待抓取的URL列表
    target_urls = [
        "https://www.myquant.cn/docs/python_strategyies/153",
        "https://www.myquant.cn/docs/python_strategyies/424",
        "https://www.myquant.cn/docs/python_strategyies/425",
        "https://www.myquant.cn/docs/python_strategyies/426",
        "https://www.myquant.cn/docs/python_strategyies/427",
        "https://www.myquant.cn/docs/python_strategyies/428",
        "https://www.myquant.cn/docs/python_strategyies/101",
        "https://www.myquant.cn/docs/python_strategyies/103",
        "https://www.myquant.cn/docs/python_strategyies/104",
        "https://www.myquant.cn/docs/python_strategyies/105",
        "https://www.myquant.cn/docs/python_strategyies/106",
        "https://www.myquant.cn/docs/python_strategyies/107",
        "https://www.myquant.cn/docs/python_strategyies/108",
        "https://www.myquant.cn/docs/python_strategyies/109",
        "https://www.myquant.cn/docs/python_strategyies/110",
        "https://www.myquant.cn/docs/python_strategyies/111",
        "https://www.myquant.cn/docs/python_strategyies/112"
    ]
    
    output_dir = "/root/autodl-tmp/FinancialStrategy2Code/datasets/fmz_strategies/myquant_strategies"
    fetcher = MyQuantStrategyFetcher(target_urls, output_dir)
    fetcher.run()

if __name__ == "__main__":
    main()