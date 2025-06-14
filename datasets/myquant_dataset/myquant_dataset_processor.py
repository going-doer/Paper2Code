#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import markdownify
import re
import os
import json
import time
import random
from pathlib import Path
import logging
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("myquant_crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StrategyCrawler:
    def __init__(self, target_urls: List[str], output_dir: str):
        """初始化爬虫"""
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
        self.strategy_dataset = []
        self.strategy_id = 1

    def fetch_webpage(self, url: str) -> str:
        """获取网页内容，带反爬延时和错误处理"""
        try:
            # 随机延时3-8秒避免请求过快
            delay = random.uniform(3, 8)
            logger.info(f"等待 {delay:.2f} 秒后请求 {url}")
            time.sleep(delay)
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            response.encoding = 'utf-8'
            logger.info(f"成功获取 {url}，状态码: {response.status_code}")
            return response.text
        except Exception as e:
            logger.error(f"请求 {url} 失败: {str(e)}")
            return ""

    def extract_strategies(self, html_content: str, url: str) -> List[Dict[str, Any]]:
        """从HTML中提取策略信息，支持多种代码块格式和内容结构"""
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        strategies = []
        page_title = soup.title.text.strip() if soup.title else "未知策略"
        page_title = re.sub(r'[\\/*?:"<>|]', '_', page_title)
        
        # 尝试多种内容区域选择器
        content_selectors = [
            'div.markdown-body', 'div.docs-content', 'main', 'article',
            'div[class*="content"]', 'div[class*="article"]', 'div[class*="post"]'
        ]
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                logger.info(f"使用选择器 '{selector}' 定位内容区域")
                break
        if not main_content:
            logger.warning(f"未找到内容区域，使用整个页面: {url}")
            main_content = soup
        
        # 转换为Markdown并提取代码块
        md_content = markdownify.markdownify(str(main_content), heading_style="atx")
        code_blocks = self._extract_code_blocks(md_content, str(main_content))
        
        # 提取描述块
        desc_blocks = self._split_description(md_content, code_blocks)
        
        # 处理每个代码块与描述的对应关系
        for i, code in enumerate(code_blocks):
            if i < len(desc_blocks):
                desc = desc_blocks[i].strip()
                strategy_title = self._extract_title(desc, page_title, i)
                
                strategy = {
                    "strategy_id": f"myquant_{self.strategy_id:03d}",
                    "title": strategy_title,
                    "description": desc,
                    "code": code.strip(),
                    "source_url": url,
                    "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tags": self._extract_tags(strategy_title, desc),
                    "metadata": {
                        "complexity": self._estimate_complexity(code),
                        "strategy_type": self._classify_strategy_type(desc),
                        "code_format": self._detect_code_format(code)
                    }
                }
                strategies.append(strategy)
                self.strategy_id += 1
        
        logger.info(f"从 {url} 提取 {len(strategies)} 个策略")
        return strategies

    def _extract_code_blocks(self, md_content: str, html_content: str) -> List[str]:
        """提取所有代码块，支持三种格式：```python、```、HTML pre标签"""
        code_blocks = []
        
        # 1. 匹配 ```python 格式
        py_blocks = re.findall(r'```python\n(.*?)\n```', md_content, re.DOTALL)
        code_blocks.extend(py_blocks)
        logger.info(f"找到 {len(py_blocks)} 个Python格式代码块")
        
        # 2. 匹配 ``` 格式（无语言标识）
        if not code_blocks:  # 如果没找到python格式，尝试通用格式
            generic_blocks = re.findall(r'```\n(.*?)\n```', md_content, re.DOTALL)
            code_blocks.extend(generic_blocks)
            logger.info(f"找到 {len(generic_blocks)} 个通用格式代码块")
        
        # 3. 匹配 HTML <pre><code> 格式
        if not code_blocks:  # 如果前面都没找到，尝试HTML格式
            html_soup = BeautifulSoup(html_content, 'html.parser')
            html_blocks = []
            for pre in html_soup.find_all('pre'):
                code = pre.find('code')
                if code:
                    code_text = code.get_text().strip()
                    # 只保留可能是策略代码的长块（至少50个字符）
                    if len(code_text) > 50:
                        html_blocks.append(code_text)
            code_blocks.extend(html_blocks)
            logger.info(f"找到 {len(html_blocks)} 个HTML格式代码块")
        
        # 过滤空块和短块
        code_blocks = [block for block in code_blocks if len(block.strip()) > 100]
        return code_blocks

    def _split_description(self, md_content: str, code_blocks: List[str]) -> List[str]:
        """将描述文本按代码块分割"""
        if not code_blocks:
            return [md_content]
        
        # 按代码块分割描述
        desc_blocks = []
        remaining_text = md_content
        
        for i, code in enumerate(code_blocks):
            # 替换代码块为标记，方便后续分割
            placeholder = f"___CODE_BLOCK_{i}___"
            remaining_text = remaining_text.replace(code, placeholder)
        
        # 按标记分割描述
        parts = remaining_text.split("___CODE_BLOCK_")
        for i, part in enumerate(parts):
            if i < len(code_blocks):
                desc = part.split("___")[0].strip()
                desc_blocks.append(desc)
        
        # 如果最后还有剩余内容，添加为最后一个描述块
        if len(parts) > len(code_blocks):
            desc_blocks.append(parts[-1].strip())
        
        return desc_blocks

    def _extract_title(self, description: str, page_title: str, index: int) -> str:
        """从描述中提取策略标题，若没有则使用页面标题"""
        # 尝试从描述中提取一级标题或二级标题
        title_match = re.search(r'^#{1,2}\s+(.*?)\n', description, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        
        # 如果没有标题，使用页面标题并添加序号
        return f"{page_title}_策略{index+1}"

    def _extract_tags(self, title: str, description: str) -> List[str]:
        """从标题和描述中提取标签"""
        tags = []
        
        # 提取括号内的内容作为标签
        bracket_tags = re.findall(r'\((.*?)\)', title)
        tags.extend(bracket_tags)
        
        # 添加关键词标签
        keywords = [
            "双均线", "MA", "MACD", "RSI", "KDJ", "BOLL", "ATR",
            "期货", "股票", "期权", "量化", "趋势跟踪", "均值回归",
            "高频", "低频", "日内", "隔夜", "多因子", "动量", "反转"
        ]
        for kw in keywords:
            if kw.lower() in title.lower() or kw.lower() in description.lower():
                tags.append(kw)
        
        # 去重并限制数量
        return list(set(tags))[:5]

    def _estimate_complexity(self, code: str) -> str:
        """根据代码行数估算复杂度"""
        lines = code.strip().split('\n')
        line_count = len(lines)
        
        if line_count < 50:
            return "simple"
        elif line_count < 150:
            return "medium"
        else:
            return "complex"

    def _classify_strategy_type(self, description: str) -> str:
        """根据描述判断策略类型"""
        desc_lower = description.lower()
        
        if any(kw in desc_lower for kw in ["趋势跟踪", "均线", "ma", "macd", "动量"]):
            return "trend_following"
        elif any(kw in desc_lower for kw in ["均值回归", "反转", "rsi", "超买超卖", "boll"]):
            return "mean_reversion"
        elif any(kw in desc_lower for kw in ["波动率", "期权", "隐含波动率", "vega"]):
            return "volatility"
        elif any(kw in desc_lower for kw in ["套利", "spread", "对冲", "配对"]):
            return "arbitrage"
        elif any(kw in desc_lower for kw in ["多因子", "因子模型", "alpha"]):
            return "multi_factor"
        else:
            return "other"

    def _detect_code_format(self, code: str) -> str:
        """检测代码格式"""
        if re.search(r'initialize\(', code):
            return "myquant_initialize"
        elif re.search(r'def\s+on_bar\(', code):
            return "myquant_on_bar"
        elif re.search(r'def\s+on_tick\(', code):
            return "myquant_on_tick"
        else:
            return "unknown"

    def save_dataset(self):
        """保存数据集为多种格式"""
        if not self.strategy_dataset:
            logger.warning("数据集为空，未保存任何文件")
            return
        
        # 1. 保存为JSONL格式（每行一个策略，适合训练）
        jsonl_path = self.output_dir / "myquant_strategies.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for strategy in self.strategy_dataset:
                f.write(json.dumps(strategy, ensure_ascii=False) + "\n")
        logger.info(f"JSONL数据集已保存至: {jsonl_path}，共 {len(self.strategy_dataset)} 个策略")
        
        # 2. 保存为JSON格式（适合预览）
        json_path = self.output_dir / "myquant_strategies.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.strategy_dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON数据集已保存至: {json_path}")
        
        # 3. 保存为CSV格式（适合分析）
        csv_path = self.output_dir / "myquant_strategies.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("strategy_id,title,source_url,complexity,strategy_type,tags\n")
            for strategy in self.strategy_dataset:
                tags = ";".join(strategy["tags"])
                f.write(f"{strategy['strategy_id']},{strategy['title']},{strategy['source_url']},"
                        f"{strategy['metadata']['complexity']},{strategy['metadata']['strategy_type']},{tags}\n")
        logger.info(f"CSV数据集已保存至: {csv_path}")
        
        # 4. 保存每个策略为单独的文件
        for strategy in self.strategy_dataset:
            # 保存代码
            code_dir = self.output_dir / "code"
            code_dir.mkdir(exist_ok=True)
            code_path = code_dir / f"{strategy['strategy_id']}.py"
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(strategy['code'])
            
            # 保存描述
            desc_dir = self.output_dir / "description"
            desc_dir.mkdir(exist_ok=True)
            desc_path = desc_dir / f"{strategy['strategy_id']}.md"
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(strategy['description'])

    def run(self):
        """执行完整抓取流程"""
        logger.info(f"开始处理 {len(self.target_urls)} 个URL...")
        for url in self.target_urls:
            html = self.fetch_webpage(url)
            strategies = self.extract_strategies(html, url)
            self.strategy_dataset.extend(strategies)
        
        self.save_dataset()
        logger.info("所有URL处理完成")


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
    
    # 输出目录
    output_dir = "/root/autodl-tmp/FinancialStrategy2Code/datasets/myquant_dataset"
    
    # 运行爬虫
    crawler = StrategyCrawler(target_urls, output_dir)
    crawler.run()


if __name__ == "__main__":
    main()