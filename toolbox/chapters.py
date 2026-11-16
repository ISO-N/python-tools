"""章节配置（``{"名称": 起始页码}``）的共享解析逻辑。

既服务于「图片按页码归档」（文件夹名有严格字符限制），也服务于
「PDF 页码 / 书签 / 水印」（章节标题允许冒号、斜杠等标点）。
"""

from __future__ import annotations

import json

INVALID_NAME_CHARS = set('<>:"/\\|?*')

AI_PROMPT = """下面我会提供书籍的目录（table of contents）图片。请从目录中提取章节标题和对应的起始页码，整理成 JSON 配置。

【提取要求】
1. 按目录中的章节标题划分文件夹，优先使用章一级标题（如“第一章”“Chapter 1”）；如果目录没有分章，则使用目录中的一级条目。
2. 文件夹名使用目录中的标题原文，去掉页码、省略号/点线、多余空白，可保留章节编号。
3. 起始页码使用目录中标注的页码，原样输出为整数；不要换算成 PDF 页码或其它编号。
4. 每个起始页码只出现一次；如果多个条目共用同一页，只保留层级更高或更主要的那个。
5. 忽略没有阿拉伯数字页码的条目。
6. 如果有多个目录图片，请合并结果后统一输出。

【输出要求】
1. 只输出一个 JSON 对象，不要 Markdown 代码块，不要任何解释文字。
2. JSON 格式为 {"文件夹名": 起始页码, ...}，按起始页码升序排列。
3. 文件夹名不能包含 / \\ : * ? " < > | 等字符。

【示例输出（页码仅为格式示意）】
{"第一章 绪论": 1, "第二章 方法": 15}"""

BOOKMARK_AI_PROMPT = """下面我会提供书籍的目录（table of contents）图片。请从目录中提取章节标题和对应的起始页码，整理成 JSON 配置。

【提取要求】
1. 按目录中的章节标题提取，优先使用章一级标题（如“第一章”“Chapter 1”）；如果目录没有分章，则使用目录中的一级条目。
2. 标题使用目录中的原文，去掉页码、省略号/点线、多余空白，可保留章节编号。
3. 起始页码使用目录中标注的页码，原样输出为整数；不要换算成 PDF 页码或其它编号。
4. 每个起始页码只出现一次；如果多个条目共用同一页，只保留层级更高或更主要的那个。
5. 忽略没有阿拉伯数字页码的条目。
6. 如果有多个目录图片，请合并结果后统一输出。

【输出要求】
1. 只输出一个 JSON 对象，不要 Markdown 代码块，不要任何解释文字。
2. JSON 格式为 {"章节标题": 起始页码, ...}，按起始页码升序排列。

【示例输出（页码仅为格式示意）】
{"第一章 绪论": 1, "第二章 方法": 15}"""


class ConfigError(ValueError):
    """JSON 配置无法解析或不符合要求时抛出。"""


def _validate_name(raw_name: str, name: str) -> None:
    if not name:
        raise ConfigError("文件夹名不能为空")
    if name in {".", ".."}:
        raise ConfigError(f"文件夹名不合法：{raw_name!r}")
    if name.endswith("."):
        raise ConfigError(f"文件夹名不能以句点结尾：{raw_name!r}")
    invalid = sorted(
        {ch for ch in name if ch in INVALID_NAME_CHARS or ord(ch) < 32}
    )
    if invalid:
        shown = " ".join(repr(ch) for ch in invalid)
        raise ConfigError(f"文件夹名 {raw_name!r} 含有不允许的字符：{shown}")


def parse_chapter_config(
    text: str,
    *,
    validate_names: bool = True,
) -> list[tuple[str, int]]:
    """解析 ``{"名称": 起始页码}``，返回按起始页码升序排列的列表。

    ``validate_names=False`` 时只校验名称非空、页码为正整数与重复情况，
    允许标题中出现文件夹不允许的标点（供 PDF 书签使用）。
    """
    label = "文件夹" if validate_names else "章节"

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigError(f"JSON 中存在重复的{label}名：{key}")
            result[key] = value
        return result

    raw = (text or "").strip()
    if not raw:
        raise ConfigError(
            f'请填写{label}配置 JSON，例如：{{"第一章": 1, "第二章": 15}}'
        )
    try:
        data = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except ConfigError:
        raise
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"JSON 语法错误（第 {exc.lineno} 行第 {exc.colno} 列）：{exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f'JSON 顶层必须是对象，格式为 {{"{label}名": 起始页码, ...}}'
        )
    if not data:
        raise ConfigError(f"JSON 配置为空，至少需要一个{label}")

    entries: list[tuple[str, int]] = []
    seen_names: set[str] = set()
    for raw_name, raw_page in data.items():
        name = raw_name.strip()
        if validate_names:
            _validate_name(raw_name, name)
        elif not name:
            raise ConfigError("章节标题不能为空")
        if name in seen_names:
            raise ConfigError(f"{label}名重复：{name!r}")
        seen_names.add(name)

        if isinstance(raw_page, bool) or not isinstance(raw_page, int):
            raise ConfigError(
                f"{label} {name!r} 的起始页码必须是整数，当前为：{raw_page!r}"
            )
        if raw_page < 1:
            raise ConfigError(
                f"{label} {name!r} 的起始页码必须大于等于 1，当前为：{raw_page}"
            )
        entries.append((name, raw_page))

    entries.sort(key=lambda item: item[1])
    for (prev_name, prev_page), (name, page) in zip(entries, entries[1:]):
        if prev_page == page:
            raise ConfigError(
                f"起始页码重复：{prev_name!r} 与 {name!r} 都使用 {page}"
            )
    return entries
