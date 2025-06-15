import os
import re
import chardet
import magic
from typing import List, Dict, Optional, Tuple

from config import (
    ENCODING_PATTERNS,
    CODE_VALIDATION_RULES,
    FMZ_API_PATTERNS,
    STRATEGY_PATTERNS
)

def detect_file_encoding(file_path: str) -> Optional[str]:
    """检测文件编码"""
    # 首先尝试使用 chardet
    with open(file_path, 'rb') as f:
        raw = f.read()
        result = chardet.detect(raw)
        if result['confidence'] > 0.8:
            return result['encoding']
    
    # 如果 chardet 检测不准确，尝试常见编码
    for encoding in ['utf-8', 'gbk', 'gb2312', 'ascii']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    
    return None

def clean_code_content(content: str) -> str:
    """清理和规范化代码内容"""
    # 移除 BOM
    content = content.replace('\ufeff', '')
    
    # 统一换行符
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # 移除多余的空行
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # 移除行尾空白
    content = '\n'.join(line.rstrip() for line in content.split('\n'))
    
    return content.strip()

def extract_code_sections(content: str) -> List[str]:
    """从文本中提取代码片段"""
    # 匹配代码块的正则表达式
    code_block_patterns = [
        r'```(?:javascript|js|python|pine)\n(.*?)\n```',  # Markdown 代码块
        r'<pre><code>(.*?)</code></pre>',  # HTML 代码块
        r'/\*\*(.*?)\*\*/',  # 多行注释
    ]
    
    sections = []
    
    # 提取各种格式的代码块
    for pattern in code_block_patterns:
        matches = re.finditer(pattern, content, re.DOTALL)
        sections.extend(match.group(1).strip() for match in matches)
    
    # 如果没有找到代码块，可能整个文件就是代码
    if not sections and any(api in content for api in FMZ_API_PATTERNS['strategy_api']):
        sections.append(content)
        
    return [clean_code_content(section) for section in sections]

def is_valid_code_section(code: str) -> bool:
    """验证代码片段是否有效"""
    # 检查长度
    if not (CODE_VALIDATION_RULES['min_length'] <= 
            len(code) <= 
            CODE_VALIDATION_RULES['max_length']):
        return False
    
    # 检查必需函数
    for func in CODE_VALIDATION_RULES['required_functions']:
        if not re.search(rf'function\s+{func}\s*\(', code):
            return False
    
    # 检查禁用关键字
    for keyword in CODE_VALIDATION_RULES['banned_keywords']:
        if keyword in code:
            return False
            
    # 检查基本语法结构
    try:
        # 简单的括号匹配检查
        if code.count('{') != code.count('}'):
            return False
        if code.count('(') != code.count(')'):
            return False
    except:
        return False
        
    return True

def merge_code_sections(sections: List[str]) -> str:
    """合并多个代码片段"""
    if not sections:
        return ""
        
    # 如果只有一个片段，直接返回
    if len(sections) == 1:
        return sections[0]
        
    # 多个片段时，按函数组织
    merged = []
    functions = {}
    
    # 提取所有函数定义
    for section in sections:
        func_matches = re.finditer(
            r'function\s+(\w+)\s*\([^)]*\)\s*{([^}]*)}',
            section,
            re.DOTALL
        )
        for match in func_matches:
            func_name = match.group(1)
            func_body = match.group(2)
            if func_name not in functions:
                functions[func_name] = func_body
    
    # 组织代码结构
    # 1. 导入和全局变量
    merged.extend(line for section in sections 
                 for line in section.split('\n')
                 if line.startswith('var') or 
                    line.startswith('let') or
                    line.startswith('const'))
    
    # 2. 函数定义
    for func_name, func_body in functions.items():
        merged.append(f"function {func_name}() {{{func_body}}}")
    
    return '\n\n'.join(merged)

def analyze_code_structure(code: str) -> Dict:
    """分析代码结构"""
    structure = {
        "functions": [],
        "api_usage": {},
        "global_vars": [],
        "complexity": {
            "lines": len(code.split('\n')),
            "functions": 0,
            "max_nesting": 0,
            "api_calls": 0
        }
    }
    
    # 分析函数
    func_matches = re.finditer(
        r'function\s+(\w+)\s*\([^)]*\)',
        code
    )
    structure["functions"] = [m.group(1) for m in func_matches]
    structure["complexity"]["functions"] = len(structure["functions"])
    
    # 分析 API 使用
    for api_type, patterns in FMZ_API_PATTERNS.items():
        structure["api_usage"][api_type] = []
        for api in patterns:
            if api in code:
                structure["api_usage"][api_type].append(api)
                structure["complexity"]["api_calls"] += 1
    
    # 分析全局变量
    var_matches = re.finditer(
        r'(?:var|let|const)\s+(\w+)\s*=',
        code
    )
    structure["global_vars"] = [m.group(1) for m in var_matches]
    
    # 分析嵌套深度
    lines = code.split('\n')
    current_depth = 0
    max_depth = 0
    for line in lines:
        current_depth += line.count('{') - line.count('}')
        max_depth = max(max_depth, current_depth)
    structure["complexity"]["max_nesting"] = max_depth
    
    return structure
