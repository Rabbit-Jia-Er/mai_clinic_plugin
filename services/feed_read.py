"""读说说服务：拉好友说说 + 评论 + 点赞 + 多层链式回复识别。"""

from __future__ import annotations

import asyncio
import logging
from ..utils import get_logger
import random
from typing import Any, Optional

from ..utils import get_global_str, peel_envelope
from .cookie import renew_cookies
from .llm_runner import LLMRunner
from .persona import resolve_persona
from .prompts import build_comment_prompt
from .qzone_api import QzoneAPI, create_qzone_api

logger = get_logger(__name__)


def _is_skip_comment(text) -> bool:
    """LLM 选择不评论时返回 True。"""
    if text is None:
        return True
    t = str(text).strip()
    if not t:
        return True
    for _ in range(2):
        t = t.strip().strip("\"'“”‘’`")
        if len(t) >= 2 and t[0] in "[(（【" and t[-1] in "])）】":
            t = t[1:-1].strip()
    normalized = "".join(t.split()).lower()
    return normalized in {
        "不回复", "不评论", "跳过", "无", "无评论",
        "无需回复", "不用回复", "不必回复",
        "skip", "none", "n/a", "na", "null",
    }


_SKIP_PROMPT_SUFFIX = (
    "。如果你觉得没必要评论（内容无感、不适合插嘴、太隐私、广告、重复水帖等），"
    "只输出三个字：不回复；否则只输出评论正文"
)


def _with_skip_prompt(prompt: str, allow_skip: bool) -> str:
    if not allow_skip:
        return prompt
    if "不回复" in prompt:
        return prompt
    return prompt.rstrip() + _SKIP_PROMPT_SUFFIX


async def renew_cookies_from_plugin(plugin) -> bool:
    """便捷封装：从 plugin 取所有参数刷 cookie。"""
    uin = await get_global_str(plugin.ctx, "bot.qq_account", "")
    return await renew_cookies(
        plugin.ctx,
        host=plugin.config.plugin.http_host,
        port=plugin.config.plugin.http_port,
        napcat_token=plugin.config.plugin.napcat_token,
        uin=uin,
        methods=list(plugin.config.plugin.cookie_methods),
    )


async def make_qzone_api(plugin) -> Optional[QzoneAPI]:
    """便捷封装：刷 cookie + 取 uin + 构造 QzoneAPI。"""
    if not await renew_cookies_from_plugin(plugin):
        return None
    uin = await get_global_str(plugin.ctx, "bot.qq_account", "")
    return create_qzone_api(uin)


# ===== 单个动作 wrapper =====


async def read_feeds(plugin, target_qq: str, num: int) -> list[dict[str, Any]]:
    """获取指定 QQ 的说说列表，失败返回 []。"""
    qzone = await make_qzone_api(plugin)
    if qzone is None:
        return []
    try:
        return await qzone.get_list(target_qq, num)
    except Exception as exc:
        logger.error("get_list 异常: %s", exc, exc_info=True)
        return []


async def comment_feed(plugin, target_qq: str, fid: str, content: str) -> bool:
    """评论指定说说。"""
    qzone = await make_qzone_api(plugin)
    if qzone is None:
        return False
    try:
        return await qzone.comment(fid, target_qq, content)
    except Exception as exc:
        logger.error("comment 异常: %s", exc, exc_info=True)
        return False


async def like_feed(plugin, target_qq: str, fid: str) -> bool:
    """点赞指定说说。"""
    qzone = await make_qzone_api(plugin)
    if qzone is None:
        return False
    try:
        return await qzone.like(fid, target_qq)
    except Exception as exc:
        logger.error("like 异常: %s", exc, exc_info=True)
        return False


async def reply_feed(
    plugin,
    fid: str,
    target_qq: str,
    target_nickname: str,
    content: str,
    comment_tid: str,
    host_uin: Optional[str] = None,
) -> bool:
    """回复指定评论。"""
    qzone = await make_qzone_api(plugin)
    if qzone is None:
        return False
    try:
        return await qzone.reply(fid, target_qq, target_nickname, content, comment_tid, host_uin=host_uin)
    except Exception as exc:
        logger.error("reply 异常: %s", exc, exc_info=True)
        return False


# ===== 高级：读说说 + 点赞评论（ReadFeed Action 用）=====


async def _get_person_info(plugin, user_id: str) -> tuple[str, str]:
    """通过 ctx.db.get(PersonInfo) 查名字和印象。失败返回 ('未知用户', '无印象')。"""
    if not user_id:
        return "未知用户", "无印象"
    try:
        result = await plugin.ctx.db.get(
            model_name="PersonInfo",
            filters={"user_id": user_id},
        )
    except Exception as exc:
        logger.debug("db.get PersonInfo 异常: %s", exc)
        return "未知用户", "无印象"
    result = peel_envelope(result)
    rows = result if isinstance(result, list) else (result.get("rows") if isinstance(result, dict) else [])
    if not rows:
        return "未知用户", "无印象"
    row = rows[0]
    name = row.get("person_name") or "未知用户"
    impression = row.get("memory_points") or "无印象"
    return str(name), str(impression)


async def read_and_engage(
    plugin,
    target_qq: str,
    target_name: str,
    num: int,
    processed_list: dict[str, list[str]],
    *,
    cache_size: int = 100,
    enable_comment: bool = True,
) -> tuple[bool, list[dict[str, Any]] | str]:
    """读说说 + 按概率点赞评论。会修改 processed_list（去重缓存）。

    Args:
        enable_comment: False 时只读/点赞，不发表评论。

    Returns:
        (success, feeds_list_or_error_message)
    """
    feeds_list = await read_feeds(plugin, target_qq, num)
    if not feeds_list:
        return False, "未读取到说说"
    first = feeds_list[0]
    if isinstance(first, dict) and first.get("error"):
        return False, str(first.get("error"))

    qzone = await make_qzone_api(plugin)
    if qzone is None:
        return False, "cookie 不存在"

    from ..utils.date import format_date_str  # noqa: F401  (保留以便扩展)

    # 印象
    _, impression = await _get_person_info(plugin, target_qq)

    # 人格（含 multiple_reply_style 抽样 + self_description）
    persona = await resolve_persona(plugin)

    runner = LLMRunner(
        plugin.ctx,
        plugin.config.llm.text_model,
        timeout=plugin.config.llm.llm_timeout_seconds,
    )
    like_p = plugin.config.read.like_probability
    comment_p = plugin.config.read.comment_probability
    allow_skip = bool(getattr(plugin.config.read, "allow_skip_comment", True))
    show_prompt = plugin.config.llm.show_prompt

    for feed in feeds_list:
        if feed.get("error"):
            continue
        fid = feed["tid"]
        if fid in processed_list:
            continue
        await asyncio.sleep(3 + random.random())

        content = feed.get("content", "")
        for image in (feed.get("images") or []):
            content = content + image
        rt_con = feed.get("rt_con", "")
        created_time = feed.get("created_time", "")

        if enable_comment and random.random() <= comment_p:
            prompt = build_comment_prompt(
                plugin, target_name, content, created_time,
                persona.personality, persona.style, impression, rt_con,
                self_description=persona.self_description,
            )
            prompt = _with_skip_prompt(prompt, allow_skip)
            if show_prompt:
                logger.info("评论 prompt: %s", prompt)
            success, comment_message = await runner.generate(prompt, temperature=0.3)
            if success and comment_message:
                if allow_skip and _is_skip_comment(comment_message):
                    logger.info("选择不评论: %s", comment_message)
                else:
                    ok = await qzone.comment(fid, target_qq, comment_message)
                    if ok:
                        logger.info("评论成功: %s", comment_message)
                    else:
                        logger.warning("评论失败")
            else:
                logger.warning("生成评论失败: %s", comment_message)
        elif not enable_comment:
            logger.info("本次读空间关闭评论，跳过说说 %s", fid)

        if random.random() <= like_p:
            ok = await qzone.like(fid, target_qq)
            if ok:
                logger.info("点赞成功: %s", content[:20])
            else:
                logger.warning("点赞失败")

        processed_list[fid] = []
        while len(processed_list) > cache_size:
            oldest = next(iter(processed_list))
            processed_list.pop(oldest)

    return True, feeds_list
