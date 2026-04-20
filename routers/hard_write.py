# nlp-gateway/routers/hard_write.py
#
# Hard Write 判定 API
#
# V2: 新增 ops 字段，统一返回所有 profile 操作（KV + 多值 + family）

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal, List, Union
import os
import logging

from services.hard_write_logic import judge_hard_write, extract_profile_ops
from services.profile_parse import parse_profile_ops

router = APIRouter()

# ---- logging ----
logger = logging.getLogger("nlp_gateway.hard_write")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False


class HardWriteJudgeReq(BaseModel):
    text: str = Field(..., min_length=1)
    mode: Literal["remember", "forget"] = Field(default="remember")
    traceId: Optional[str] = None


# ---- Legacy ProfileOp (for backward compatibility) ----
class ProfileOp(BaseModel):
    field: Literal["likes", "ng"]
    op: Literal["add"]
    items: List[str]


# ---- V2: Unified Op schema ----
class ProfileOpV2(BaseModel):
    """
    统一的 profile 操作格式

    Examples:
        {"target": "profile", "field": "name", "op": "set", "value": "小明"}
        {"target": "profile", "field": "likes", "op": "add", "items": ["猫", "狗"]}
        {"target": "profile", "field": "family", "op": "merge", "patch": {"has_siblings": true}}
        {"target": "profile", "field": "birthday_ymd", "op": "clear"}
        {"target": "plan", "content": "明天开会"}
        {"target": "memory", "content": "..."}
    """
    target: Literal["profile", "plan", "memory"]
    field: Optional[str] = None  # profile 字段名
    op: Optional[Literal["set", "clear", "add", "remove", "merge", "clear_all"]] = None
    value: Optional[str] = None  # set 时的值
    items: Optional[List[str]] = None  # add/remove 时的列表
    patch: Optional[Dict[str, Any]] = None  # merge 时的补丁（family）
    content: Optional[str] = None  # plan/memory 时的内容


class HardWriteJudgeResp(BaseModel):
    form_ok: bool
    target: Literal["plan", "profile", "memory", "block"]
    confidence: float
    reason: List[str]
    profile_anchor_hit: bool
    # Legacy field (backward compatible)
    profile_ops: Optional[List[ProfileOp]] = None
    # V2: Unified ops list
    ops: Optional[List[ProfileOpV2]] = None


@router.post("/judge", response_model=HardWriteJudgeResp)
async def hard_write_judge(req: HardWriteJudgeReq) -> Dict[str, Any]:
    """
    判定硬写入内容的目标类型

    - form_ok: 内容形式上是否可记忆
    - target: plan | profile | memory | block
    - confidence: 置信度（v1 规则版固定 0.8）
    - reason: 判定理由列表
    - profile_ops: (legacy) likes/ng 操作列表
    - ops: (v2) 统一的操作列表，包含所有 profile KV/多值/family 操作
    """
    trace_id = req.traceId or "-"
    logger.info("[traceId=%s] /hard_write/judge mode=%s text=%s", trace_id, req.mode, req.text[:60] if req.text else "")

    result = judge_hard_write(text=req.text, mode=req.mode)

    # ========== V2: 使用 parse_profile_ops 生成统一 ops ==========
    ops_v2 = None
    profile_ops_legacy = None
    likes_n = 0
    ngs_n = 0

    # 条件：form_ok 且 (target=profile 或 profile_anchor_hit)
    # forget mode 也需要解析（用于 remove 操作）
    should_parse = (
        result["form_ok"]
        and (result["target"] == "profile" or result.get("profile_anchor_hit"))
    )

    if should_parse:
        try:
            # V2: 统一解析
            parsed_ops = parse_profile_ops(req.text, mode=req.mode, lang="zh")

            if parsed_ops:
                ops_v2 = parsed_ops

                # 统计
                for op in parsed_ops:
                    if op.get("field") == "likes" and op.get("items"):
                        likes_n += len(op["items"])
                    elif op.get("field") == "ng" and op.get("items"):
                        ngs_n += len(op["items"])

            # Legacy: 同时生成旧格式的 profile_ops（仅 likes/ng，仅 remember+add）
            if req.mode == "remember" and parsed_ops:
                legacy_list = []
                for op in parsed_ops:
                    if op.get("field") == "likes" and op.get("op") == "add" and op.get("items"):
                        legacy_list.append({
                            "field": "likes",
                            "op": "add",
                            "items": op["items"]
                        })
                    elif op.get("field") == "ng" and op.get("op") == "add" and op.get("items"):
                        legacy_list.append({
                            "field": "ng",
                            "op": "add",
                            "items": op["items"]
                        })
                if legacy_list:
                    profile_ops_legacy = legacy_list

        except Exception as e:
            logger.warning("[traceId=%s] /hard_write/judge parse_profile_ops failed: %s", trace_id, e)

            # Fallback: 使用旧的 extract_profile_ops（仅 likes/ng）
            if req.mode == "remember":
                try:
                    ops_result = extract_profile_ops(req.text)
                    legacy_list = []

                    if ops_result.get("likes"):
                        legacy_list.append({
                            "field": "likes",
                            "op": "add",
                            "items": ops_result["likes"]
                        })
                        likes_n = len(ops_result["likes"])

                    if ops_result.get("ngs"):
                        legacy_list.append({
                            "field": "ng",
                            "op": "add",
                            "items": ops_result["ngs"]
                        })
                        ngs_n = len(ops_result["ngs"])

                    if legacy_list:
                        profile_ops_legacy = legacy_list
                        # 也填充 ops_v2
                        ops_v2 = [
                            {"target": "profile", "field": op["field"], "op": "add", "items": op["items"]}
                            for op in legacy_list
                        ]

                except Exception as e2:
                    logger.warning("[traceId=%s] /hard_write/judge fallback extract_profile_ops failed: %s", trace_id, e2)

    result["profile_ops"] = profile_ops_legacy
    result["ops"] = ops_v2

    logger.info("[traceId=%s] /hard_write/judge => form_ok=%s target=%s profile_anchor_hit=%s likes_n=%d ngs_n=%d ops_count=%d",
                trace_id, result["form_ok"], result["target"], result.get("profile_anchor_hit"),
                likes_n, ngs_n, len(ops_v2) if ops_v2 else 0)

    # ===== DEBUG: 返回前打印完整响应 =====
    print(f"\n{'='*60}")
    print(f"[DEBUG] /hard_write/judge 返回:")
    print(f"  text: {req.text}")
    print(f"  mode: {req.mode}")
    print(f"  form_ok: {result['form_ok']}")
    print(f"  target: {result['target']}")
    print(f"  profile_anchor_hit: {result.get('profile_anchor_hit')}")
    print(f"  reason: {result.get('reason')}")
    if ops_v2:
        print(f"  ops ({len(ops_v2)}):")
        for i, op in enumerate(ops_v2):
            print(f"    [{i}] {op}")
    else:
        print(f"  ops: None")
    print(f"{'='*60}\n")

    return result
