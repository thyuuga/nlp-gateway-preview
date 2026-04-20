# nlp-gateway/services/profile_parse.py
#
# Profile 解析模块 - 基于 NLP 的用户画像指令解析
#
# 设计原则：
#   - 使用 HanLP 分词，避免大量正则罗列
#   - 锚点词表结构支持多语言扩展
#   - 支持复合句（如 "记住我叫小明，生日是12月16日"）
#
# 输出格式：ops 列表
#   [
#     {"target": "profile", "field": "name", "op": "set", "value": "小明"},
#     {"target": "profile", "field": "birthday_ymd", "op": "set", "value": "0000-12-16"},
#     {"target": "profile", "field": "likes", "op": "add", "items": ["猫", "狗"]},
#   ]

import re
import logging
import os
from typing import List, Dict, Any, Optional, Literal, Tuple

# ---- logging ----
logger = logging.getLogger("nlp_gateway.profile_parse")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False


# ============================================================================
# 锚点词表 (Anchor Words) - 多语言可扩展结构
# ============================================================================

# 每个字段的锚点词，按优先级排序（长词优先）
# 格式: { field: { lang: [anchor_words] } }
ANCHOR_WORDS = {
    # ---- Profile KV 字段 ----
    "name": {
        "zh": ["我的名字是", "我的名字叫", "我叫", "名字是", "名字叫"],
    },
    # addressing: 明确指示 AI 使用的称呼 → addressing_name (set)
    "addressing": {
        "zh": ["以后叫我", "以后都叫我", "你可以叫我", "就叫我", "请叫我", "叫我"],
    },
    # nickname: 分享昵称信息 → nicknames (add)
    "nickname": {
        "zh": ["我的昵称是", "我的昵称叫", "我的小名是", "我的小名叫", "我的外号是", "我的外号叫",
               "他们都叫我", "他们叫我", "大家都叫我", "大家叫我", "朋友都叫我", "朋友叫我"],
    },
    "birthday": {
        "zh": ["我的生日是", "我生日是", "生日是", "我的生日", "我生日", "生日"],
    },
    "hometown": {
        "zh": ["故乡是", "老家是", "家乡是", "故乡在", "老家在", "家乡在", "故乡", "老家", "家乡",
               "来自", "我来自", "我从"],
    },
    "residence": {
        "zh": ["我现在住在", "现在住在", "目前住在", "我住在", "住在", "住址是",
               "我现在在", "现在在", "目前在"],
    },
    "occupation": {
        "zh": ["我的职业是", "我的工作是", "职业是", "工作是", "是做"],
    },
    "language": {
        "zh": ["以后用"],  # "以后用中文跟我说"
    },
    # ---- Profile 多值字段 ----
    "likes": {
        "zh": ["很喜欢", "特别喜欢", "超喜欢", "喜欢", "爱", "最爱", "超爱"],
    },
    "ng": {
        "zh": ["不喜欢", "讨厌", "雷点是", "是雷点", "雷点", "不要提", "别提", "不要聊", "别聊", "恨"],
    },
    # ---- Family ----
    "family": {
        "zh": ["有兄弟姐妹", "没有兄弟姐妹", "有一个哥哥", "有一个姐姐", "有一个弟弟", "有一个妹妹",
               "有两个哥哥", "有两个姐姐", "有两个弟弟", "有两个妹妹",
               "有哥哥", "有姐姐", "有弟弟", "有妹妹"],
    },
}

# 遗忘锚点词（用于 forget mode）
FORGET_ANCHOR_WORDS = {
    "name": {
        "zh": ["我的名字", "名字"],
    },
    "birthday": {
        "zh": ["我的生日", "生日"],
    },
    "hometown": {
        "zh": ["我的故乡", "我的老家", "我的家乡", "故乡", "老家", "家乡"],
    },
    "residence": {
        "zh": ["我的住址", "住址", "我住哪", "住哪"],
    },
    "occupation": {
        "zh": ["我的职业", "职业"],
    },
    "likes": {
        "zh": ["我喜欢", "我很喜欢", "喜欢"],
    },
    "ng": {
        "zh": ["我讨厌", "讨厌"],
    },
    "nicknames": {
        "zh": ["我的昵称", "我的小名", "我的外号", "昵称", "小名", "外号", "所有昵称", "所有小名", "所有外号"],
    },
}

# 职业词表（用于 "记住我是程序员" 这类没有明确锚点的句式）
OCCUPATION_WORDS = {
    "zh": [
        "程序员", "工程师", "设计师", "医生", "护士", "老师", "教师", "教授", "学生",
        "律师", "会计", "销售", "经理", "总监", "产品经理", "运营", "编辑", "记者",
        "作家", "画家", "音乐家", "演员", "导演", "厨师", "司机", "警察", "军人",
        "公务员", "自由职业者", "创业者", "企业家", "研究员", "科学家", "分析师",
        "顾问", "咨询师", "翻译", "摄影师", "建筑师", "药剂师", "兽医", "飞行员",
        "空姐", "模特", "运动员", "教练", "保姆", "快递员", "外卖员", "网红", "主播", "博主", "UP主",
    ],
}

# 语言词表
LANGUAGE_WORDS = {
    "zh": {
        "中文": "zh-CN",
        "日语": "ja",
        "日文": "ja",
        "英语": "en",
        "英文": "en",
    },
}


# ============================================================================
# HanLP 分词 (Lazy Load)
# ============================================================================

_hanlp_tok = None


def _get_hanlp_tok():
    """Lazy load HanLP tokenizer"""
    global _hanlp_tok
    if _hanlp_tok is None:
        try:
            import hanlp
            # 只加载分词，不需要完整 pipeline
            _hanlp_tok = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
            logger.info("HanLP tokenizer loaded successfully")
        except Exception as e:
            logger.warning("Failed to load HanLP tokenizer: %s", e)
            _hanlp_tok = "FAILED"
    return _hanlp_tok if _hanlp_tok != "FAILED" else None


def tokenize(text: str, lang: str = "zh") -> List[str]:
    """
    分词

    Args:
        text: 待分词文本
        lang: 语言代码 (目前只支持 zh)

    Returns:
        分词结果列表
    """
    if not text:
        return []

    if lang == "zh":
        tok = _get_hanlp_tok()
        if tok is not None:
            try:
                tokens = tok(text)
                logger.debug("tokenize: %s -> %s", text[:30], tokens)
                return tokens
            except Exception as e:
                logger.warning("HanLP tokenize failed: %s", e)

        # Fallback: 简单按字符分割（保留连续数字/英文）
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+|[^\s\u4e00-\u9fff\w]+', text)
        return tokens

    # 其他语言：暂时按空格分割
    return text.split()


# ============================================================================
# 锚点扫描 + 分段
# ============================================================================

def _build_anchor_index(lang: str = "zh") -> List[Tuple[str, str]]:
    """
    构建锚点索引：按长度降序排列的 (anchor_word, field) 列表
    """
    index = []
    for field, lang_dict in ANCHOR_WORDS.items():
        anchors = lang_dict.get(lang, [])
        for anchor in anchors:
            index.append((anchor, field))
    # 按长度降序排列，确保长词优先匹配
    index.sort(key=lambda x: len(x[0]), reverse=True)
    return index


def _build_forget_anchor_index(lang: str = "zh") -> List[Tuple[str, str]]:
    """构建遗忘锚点索引"""
    index = []
    for field, lang_dict in FORGET_ANCHOR_WORDS.items():
        anchors = lang_dict.get(lang, [])
        for anchor in anchors:
            index.append((anchor, field))
    index.sort(key=lambda x: len(x[0]), reverse=True)
    return index


# 缓存锚点索引
_ANCHOR_INDEX_CACHE: Dict[str, List[Tuple[str, str]]] = {}
_FORGET_ANCHOR_INDEX_CACHE: Dict[str, List[Tuple[str, str]]] = {}


def get_anchor_index(lang: str = "zh") -> List[Tuple[str, str]]:
    if lang not in _ANCHOR_INDEX_CACHE:
        _ANCHOR_INDEX_CACHE[lang] = _build_anchor_index(lang)
    return _ANCHOR_INDEX_CACHE[lang]


def get_forget_anchor_index(lang: str = "zh") -> List[Tuple[str, str]]:
    if lang not in _FORGET_ANCHOR_INDEX_CACHE:
        _FORGET_ANCHOR_INDEX_CACHE[lang] = _build_forget_anchor_index(lang)
    return _FORGET_ANCHOR_INDEX_CACHE[lang]


def find_anchors(text: str, mode: Literal["remember", "forget"] = "remember", lang: str = "zh") -> List[Dict]:
    """
    在文本中查找所有锚点及其位置

    Returns:
        [{"field": str, "anchor": str, "start": int, "end": int}, ...]
    """
    if not text:
        return []

    index = get_anchor_index(lang) if mode == "remember" else get_forget_anchor_index(lang)
    results = []
    used_ranges = []  # 已使用的范围，避免重叠

    for anchor, field in index:
        start = 0
        while True:
            pos = text.find(anchor, start)
            if pos == -1:
                break

            end = pos + len(anchor)

            # 检查是否与已有范围重叠
            overlaps = False
            for used_start, used_end in used_ranges:
                if not (end <= used_start or pos >= used_end):
                    overlaps = True
                    break

            if not overlaps:
                results.append({
                    "field": field,
                    "anchor": anchor,
                    "start": pos,
                    "end": end,
                })
                used_ranges.append((pos, end))

            start = pos + 1

    # 按位置排序
    results.sort(key=lambda x: x["start"])
    return results


# 局部遗忘关键词（在段落前出现时覆盖全局 mode）
LOCAL_FORGET_KEYWORDS = ["忘记", "忘了", "忘掉", "别记", "不要记"]


def _detect_local_mode(text: str, anchor_start: int) -> Optional[str]:
    """
    检测锚点前是否有局部模式覆盖词

    Args:
        text: 完整文本
        anchor_start: 锚点在文本中的起始位置

    Returns:
        "forget" 如果检测到局部遗忘关键词，否则 None
    """
    # 检查锚点前10个字符内是否有忘记关键词
    look_back = 10
    prefix_start = max(0, anchor_start - look_back)
    prefix_text = text[prefix_start:anchor_start]

    for kw in LOCAL_FORGET_KEYWORDS:
        if kw in prefix_text:
            return "forget"

    return None


def _find_local_forget_boundary(value_text: str) -> Optional[int]:
    """
    在 value_text 中查找局部遗忘关键词的位置

    Returns:
        关键词的起始位置，如果没有则返回 None
    """
    for kw in LOCAL_FORGET_KEYWORDS:
        pos = value_text.find(kw)
        if pos != -1:
            return pos
    return None


def segment_by_anchors(text: str, anchors: List[Dict]) -> List[Dict]:
    """
    按锚点分段

    Returns:
        [{"field": str, "anchor": str, "value_text": str, "start": int, "end": int, "local_mode": str|None}, ...]
    """
    if not anchors:
        return []

    segments = []
    for i, anchor in enumerate(anchors):
        # value_text 是锚点后到下一个锚点前的文本
        value_start = anchor["end"]
        if i + 1 < len(anchors):
            value_end = anchors[i + 1]["start"]
        else:
            value_end = len(text)

        value_text = text[value_start:value_end].strip()
        # 去掉开头的标点
        value_text = re.sub(r'^[，,。.、：:；;\s]+', '', value_text)
        # 去掉结尾的标点（但保留可能是内容的部分）
        value_text = re.sub(r'[，,。.、：:；;\s]+$', '', value_text)
        # 去掉结尾的 "，我" / "、我" 等模式（下一句的主语残留）
        value_text = re.sub(r'[，,。.、；;\s]+我$', '', value_text)

        # 检测局部模式覆盖
        local_mode = _detect_local_mode(text, anchor["start"])

        # 如果当前段没有 local_mode，检查 value_text 中是否有局部遗忘关键词
        # 如果有，截断 value_text 到该关键词之前
        if not local_mode:
            forget_boundary = _find_local_forget_boundary(value_text)
            if forget_boundary is not None and forget_boundary > 0:
                # 截断到遗忘关键词之前
                value_text = value_text[:forget_boundary].strip()
                # 去掉结尾的连接词和标点
                value_text = re.sub(r'[，,。.、：:；;还有和与以及并且\s]+$', '', value_text)

        segments.append({
            "field": anchor["field"],
            "anchor": anchor["anchor"],
            "value_text": value_text,
            "start": anchor["start"],
            "end": value_end,
            "local_mode": local_mode,
        })

    return segments


# ============================================================================
# 值抽取器 (Value Extractors)
# ============================================================================

def extract_name_value(value_text: str, tokens: List[str], lang: str = "zh") -> Optional[str]:
    """抽取名字"""
    if not value_text:
        return None

    text = value_text

    # 名字不应包含逗号/句号，在这些分隔符处截断
    for sep in ['，', ',', '。', '.', '；', ';', '！', '!', '？', '?']:
        if sep in text:
            text = text.split(sep)[0]

    # 去掉尾部助词
    value = re.sub(r'[吧呢啊呀哦噢嗯了]+$', '', text).strip()
    if value:
        return value
    return None


def extract_nickname_value(value_text: str, tokens: List[str], lang: str = "zh") -> Optional[str]:
    """抽取昵称"""
    if not value_text:
        return None

    text = value_text

    # 昵称不应包含逗号/句号，在这些分隔符处截断
    for sep in ['，', ',', '。', '.', '；', ';', '！', '!', '？', '?']:
        if sep in text:
            text = text.split(sep)[0]

    # 去掉尾部语气词
    value = re.sub(r'[吧呢啊呀哦噢嗯了好了可以吗]+$', '', text).strip()
    if value:
        return value
    return None


def extract_birthday_value(value_text: str, tokens: List[str], lang: str = "zh") -> Optional[str]:
    """
    抽取生日，返回 YYYY-MM-DD 格式
    支持：1993年12月16日、12月16日、1993-12-16、12-16
    """
    if not value_text:
        return None

    # 中文格式：YYYY年MM月DD日
    m = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})[日号]?', value_text)
    if m:
        year = m.group(1)
        month = m.group(2).zfill(2)
        day = m.group(3).zfill(2)
        return f"{year}-{month}-{day}"

    # 中文格式：MM月DD日（无年份）
    m = re.search(r'(\d{1,2})月(\d{1,2})[日号]?', value_text)
    if m:
        month = m.group(1).zfill(2)
        day = m.group(2).zfill(2)
        return f"0000-{month}-{day}"

    # ISO 格式：YYYY-MM-DD
    m = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', value_text)
    if m:
        year = m.group(1)
        month = m.group(2).zfill(2)
        day = m.group(3).zfill(2)
        return f"{year}-{month}-{day}"

    # 简化格式：MM-DD
    m = re.search(r'(\d{1,2})-(\d{1,2})', value_text)
    if m:
        month = m.group(1).zfill(2)
        day = m.group(2).zfill(2)
        return f"0000-{month}-{day}"

    return None


def extract_location_value(value_text: str, tokens: List[str], lang: str = "zh") -> Optional[str]:
    """抽取地点（故乡/住址）"""
    if not value_text:
        return None

    text = value_text

    # 地点不应包含逗号/句号等，在这些分隔符处截断
    for sep in ['，', ',', '。', '.', '；', ';', '！', '!', '？', '?']:
        if sep in text:
            text = text.split(sep)[0]

    # 去掉尾部语气词和方向词
    value = re.sub(r'[吧呢啊呀哦噢嗯了来去]+$', '', text).strip()
    if value:
        return value
    return None


def extract_occupation_value(value_text: str, tokens: List[str], lang: str = "zh") -> Optional[str]:
    """抽取职业"""
    if not value_text:
        return None

    text = value_text

    # 职业不应包含逗号/句号等，在这些分隔符处截断
    for sep in ['，', ',', '。', '.', '；', ';', '！', '!', '？', '?']:
        if sep in text:
            text = text.split(sep)[0]

    # 去掉 "的" 结尾（"是做XX的"）
    value = re.sub(r'的$', '', text).strip()
    # 去掉语气词
    value = re.sub(r'[吧呢啊呀哦噢嗯了]+$', '', value).strip()

    if value:
        return value
    return None


def extract_language_value(value_text: str, tokens: List[str], lang: str = "zh") -> Optional[str]:
    """抽取语言偏好，返回语言代码"""
    if not value_text:
        return None

    lang_map = LANGUAGE_WORDS.get(lang, {})
    for word, code in lang_map.items():
        if word in value_text:
            return code

    return None


def extract_likes_items(value_text: str, tokens: List[str], lang: str = "zh") -> List[str]:
    """
    抽取喜好项目列表
    集成 hard_write_logic 中的 HanLP 分词逻辑
    """
    if not value_text:
        return []

    try:
        # 尝试使用 hard_write_logic 的完整抽取逻辑
        from services.hard_write_logic import _tokenize_with_pos, _merge_noun_phrases, _extract_vo_objects, _clean_items

        tokens_pos = _tokenize_with_pos(value_text)
        if tokens_pos:
            # 合并名词短语
            noun_phrases = _merge_noun_phrases(tokens_pos)
            # V-O 规则提取
            vo_objects = _extract_vo_objects(tokens_pos)
            # 合并并清洗
            all_items = noun_phrases + vo_objects
            if all_items:
                return _clean_items(all_items, max_count=10)

    except Exception as e:
        logger.debug("extract_likes_items: fallback due to %s", e)

    # Fallback: 简单分割
    items = re.split(r'[，,、和与跟以及还有并且]+', value_text)
    result = []
    for item in items:
        item = item.strip()
        # 去掉语气词
        item = re.sub(r'[吧呢啊呀哦噢嗯了]+$', '', item).strip()
        if item and len(item) >= 1:
            result.append(item)

    return result


def extract_ng_items(value_text: str, tokens: List[str], lang: str = "zh") -> List[str]:
    """抽取雷点项目列表"""
    # 逻辑同 likes
    return extract_likes_items(value_text, tokens, lang)


def extract_family_value(value_text: str, anchor: str, lang: str = "zh") -> Optional[Dict]:
    """
    抽取家庭信息
    返回 patch 对象，如 {"has_siblings": True, "older_brother": 1}
    """
    # 根据锚点词判断
    if "没有兄弟姐妹" in anchor:
        return {"has_siblings": False, "siblings_count": 0}

    if "有兄弟姐妹" in anchor:
        return {"has_siblings": True}

    # 具体兄弟姐妹
    sibling_map = {
        "哥哥": "older_brother",
        "姐姐": "older_sister",
        "弟弟": "younger_brother",
        "妹妹": "younger_sister",
    }

    count_map = {
        "一个": 1, "两个": 2, "三个": 3, "四个": 4, "五个": 5,
        "1个": 1, "2个": 2, "3个": 3, "4个": 4, "5个": 5,
    }

    for sibling_word, field_key in sibling_map.items():
        if sibling_word in anchor:
            count = 1
            for count_word, count_val in count_map.items():
                if count_word in anchor:
                    count = count_val
                    break
            return {"has_siblings": True, field_key: count}

    return None


# ============================================================================
# 主解析函数
# ============================================================================

def parse_profile_ops(
    text: str,
    mode: Literal["remember", "forget"] = "remember",
    lang: str = "zh"
) -> List[Dict[str, Any]]:
    """
    解析用户消息，返回 profile 操作列表

    Args:
        text: 用户消息（已去掉指令词前缀，如 "记住"）
        mode: remember | forget
        lang: 语言代码

    Returns:
        ops 列表，每个 op 格式：
        - KV 字段: {"target": "profile", "field": str, "op": "set"|"clear", "value": str}
        - 多值字段: {"target": "profile", "field": str, "op": "add"|"remove", "items": List[str]}
        - Family: {"target": "profile", "field": "family", "op": "merge", "patch": Dict}
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    logger.debug("parse_profile_ops: mode=%s text=%s", mode, text[:60])

    # 1. 分词
    tokens = tokenize(text, lang)
    logger.debug("parse_profile_ops: tokens=%s", tokens)

    # 2. 查找锚点
    anchors = find_anchors(text, mode, lang)
    logger.debug("parse_profile_ops: anchors=%s", anchors)

    if not anchors:
        # 没有锚点，尝试特殊句式
        ops = _try_special_patterns(text, mode, lang)
        return ops

    # 3. 按锚点分段
    segments = segment_by_anchors(text, anchors)
    logger.debug("parse_profile_ops: segments=%s", segments)

    # 4. 对每个段抽取值
    ops = []
    for seg in segments:
        field = seg["field"]
        value_text = seg["value_text"]
        anchor = seg["anchor"]
        local_mode = seg.get("local_mode")
        # 使用局部模式覆盖（如果有的话）
        seg_mode = local_mode or mode

        if local_mode:
            logger.debug("parse_profile_ops: local_mode override: %s -> %s for field=%s", mode, local_mode, field)

        op = _extract_op_from_segment(field, value_text, anchor, tokens, seg_mode, lang)
        if op:
            ops.append(op)

    logger.debug("parse_profile_ops: result ops=%s", ops)
    return ops


def _extract_op_from_segment(
    field: str,
    value_text: str,
    anchor: str,
    tokens: List[str],
    mode: str,
    lang: str
) -> Optional[Dict]:
    """从单个段落抽取 op"""

    if mode == "forget":
        # 遗忘模式
        if field in ("likes", "ng"):
            items = extract_likes_items(value_text, tokens, lang) if field == "likes" else extract_ng_items(value_text, tokens, lang)
            if items:
                return {"target": "profile", "field": field, "op": "remove", "items": items}
            else:
                # 没有具体项目，可能是 "忘了我喜欢的" 这种，返回 clear_all
                return {"target": "profile", "field": field, "op": "clear_all"}
        elif field == "nicknames":
            # nicknames 是多值字段，清空所有昵称
            return {"target": "profile", "field": "nicknames", "op": "clear_all"}
        else:
            # KV 字段清除
            return {"target": "profile", "field": field, "op": "clear"}

    # 记忆模式
    if field == "name":
        value = extract_name_value(value_text, tokens, lang)
        if value:
            return {"target": "profile", "field": "name", "op": "set", "value": value}

    elif field == "addressing":
        # "以后叫我X" → 直接设置 addressing_name
        value = extract_nickname_value(value_text, tokens, lang)
        if value:
            return {"target": "profile", "field": "addressing_name", "op": "set", "value": value}

    elif field == "nickname":
        # "我的昵称/小名/外号是X" → 添加到 nicknames 列表
        value = extract_nickname_value(value_text, tokens, lang)
        if value:
            return {"target": "profile", "field": "nicknames", "op": "add", "items": [value]}

    elif field == "birthday":
        value = extract_birthday_value(value_text, tokens, lang)
        if value:
            return {"target": "profile", "field": "birthday_ymd", "op": "set", "value": value}

    elif field == "hometown":
        value = extract_location_value(value_text, tokens, lang)
        if value:
            return {"target": "profile", "field": "hometown", "op": "set", "value": value}

    elif field == "residence":
        value = extract_location_value(value_text, tokens, lang)
        if value:
            return {"target": "profile", "field": "residence", "op": "set", "value": value}

    elif field == "occupation":
        value = extract_occupation_value(value_text, tokens, lang)
        if value:
            return {"target": "profile", "field": "occupation", "op": "set", "value": value}

    elif field == "language":
        value = extract_language_value(value_text, tokens, lang)
        if value:
            return {"target": "profile", "field": "preferred_language", "op": "set", "value": value}

    elif field == "likes":
        items = extract_likes_items(value_text, tokens, lang)
        if items:
            return {"target": "profile", "field": "likes", "op": "add", "items": items}

    elif field == "ng":
        items = extract_ng_items(value_text, tokens, lang)
        if items:
            return {"target": "profile", "field": "ng", "op": "add", "items": items}

    elif field == "family":
        patch = extract_family_value(value_text, anchor, lang)
        if patch:
            return {"target": "profile", "field": "family", "op": "merge", "patch": patch}

    return None


def _try_special_patterns(text: str, mode: str, lang: str) -> List[Dict]:
    """
    尝试特殊句式匹配（无明确锚点的情况）
    如 "记住我是程序员"
    """
    ops = []

    if mode == "remember" and lang == "zh":
        # "我是程序员" / "我是一名医生"
        m = re.search(r'我是(?:一名|一个)?\s*(.+)$', text)
        if m:
            value = m.group(1).strip()
            # 检查是否是职业词
            occupation_list = OCCUPATION_WORDS.get(lang, [])
            for occ in occupation_list:
                if value == occ or value.endswith(occ):
                    ops.append({
                        "target": "profile",
                        "field": "occupation",
                        "op": "set",
                        "value": occ
                    })
                    return ops

    return ops


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    # 简单测试
    test_cases = [
        ("我叫小明", "remember"),
        ("我叫小明，生日是12月16日", "remember"),
        ("我叫小明生日是1993年12月16日喜欢猫", "remember"),
        ("喜欢猫和狗但讨厌蛇", "remember"),
        ("我住在东京", "remember"),
        ("我的故乡是北京", "remember"),
        ("我是程序员", "remember"),
        ("以后用日语跟我说", "remember"),
        # addressing vs nickname 测试
        ("以后叫我阿明", "remember"),           # → addressing_name
        ("你可以叫我hyuuga", "remember"),      # → addressing_name
        ("叫我小明吧", "remember"),              # → addressing_name
        ("我的昵称是阿明", "remember"),          # → nicknames
        ("他们都叫我小明", "remember"),          # → nicknames
        ("我的小名叫明明", "remember"),          # → nicknames
        ("我有一个哥哥", "remember"),
        ("我的生日", "forget"),
        ("我喜欢猫", "forget"),
        # 混合模式测试：全局 remember 但局部 forget
        ("我喜欢猫和狗狗还有忘记我讨厌香菜", "remember"),  # likes add + ng remove
        ("喜欢吃苹果，忘了我讨厌西瓜", "remember"),         # likes add + ng remove
    ]

    for text, mode in test_cases:
        ops = parse_profile_ops(text, mode)
        print(f"\n[{mode}] {text}")
        for op in ops:
            print(f"  -> {op}")
