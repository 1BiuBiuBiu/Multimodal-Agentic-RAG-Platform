"""MinerU 2.x CLI adapter with separately extracted, persistent PDF elements."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

from ingestion.native_symbols import recover_table_symbols


class MinerUError(RuntimeError):
    """An actionable parser/deployment error."""


class MinerUUnavailable(MinerUError):
    pass


_PARSE_LOCK = RLock()
NORMALIZATION_VERSION = 7
_CAPTION = re.compile(r"^\s*(Table|Figure|Fig\.?|图|表)\s*(\d+(?:[.-]\d+)*)\s*[:：.]", re.I)


def _caption_key(text):
    match = _CAPTION.match(text)
    if not match:
        return None
    return ("table" if match[1].lower() in ("table", "表") else "image", match[2])


def _associate_captions(result, positions):
    """Correct detached/swapped captions using their physical page positions."""
    pool = {}
    for position in positions:
        key = _caption_key(position['text'])
        if key:
            pool[(position['page'], *key)] = position
    for element in result['body']:
        key = _caption_key(element['text'])
        if key and element.get('bbox'):
            pool.setdefault((element['page'], *key), element)
    removed = set()
    table_owners = []
    for (page, kind, label), caption in pool.items():
        bbox = caption.get('bbox')
        if not bbox:
            continue
        candidates = []
        for element in result['tables' if kind == 'table' else 'images']:
            box = element.get('bbox')
            if element['page'] != page or not box:
                continue
            overlap = max(0, min(bbox[2], box[2]) - max(bbox[0], box[0]))
            if overlap < 0.3 * min(bbox[2]-bbox[0], box[2]-box[0]):
                continue
            # Prefer table captions above, figure captions below. Still allow
            # the opposite placement when no nearby owner exists on that side.
            above, below = box[1]-bbox[3], bbox[1]-box[3]
            score = min(abs(above) + (0 if kind == 'table' and above >= -3 else 40),
                        abs(below) + (0 if kind == 'image' and below >= -3 else 40))
            candidates.append((score, element))
        if not candidates:
            continue
        score, owner = min(candidates, key=lambda pair: pair[0])
        if score > 180:
            continue
        # Reuse the existing formatted text when this label was nested under a
        # different owner; keep non-label panel descriptions intact.
        value = caption['text']
        for element in result['tables' if kind == 'table' else 'images']:
            captions = element['caption']
            if isinstance(captions, str):
                captions = [captions]
            kept = []
            for text in captions:
                if element['page'] == page and _caption_key(text) == (kind, label):
                    value = text
                else:
                    kept.append(text)
            element['caption'] = kept
        owner['caption'].append(value)
        owner.setdefault('caption_bboxes', []).append(bbox)
        if kind == 'table':
            table_owners.append((owner, bbox))
        for element in result['body']:
            if element['page'] == page and _caption_key(element['text']) == (kind, label):
                removed.add(element['element_id'])
    if removed:
        result['body'] = [e for e in result['body'] if e['element_id'] not in removed]
        result['elements'] = [e for e in result['elements'] if e['element_id'] not in removed]
    # One full-width caption can describe aligned left/right parts of one
    # logical table. Keep those parts together instead of indexing two chunks.
    for owner, caption_box in table_owners:
        if owner not in result['tables']:
            continue
        box = owner['bbox']
        peers = []
        for table in result['tables']:
            other = table.get('bbox')
            if table is owner or table['page'] != owner['page'] or table['caption'] or not other:
                continue
            vertical = max(0, min(box[3],other[3])-max(box[1],other[1]))
            horizontal = max(0, min(box[2],other[2])-max(box[0],other[0]))
            if (abs(box[1]-other[1]) <= 15
                    and vertical >= .7*min(box[3]-box[1],other[3]-other[1])
                    and horizontal < .1*min(box[2]-box[0],other[2]-other[0])
                    and caption_box[0] <= (other[0]+other[2])/2 <= caption_box[2]
                    and caption_box[0] <= (box[0]+box[2])/2 <= caption_box[2]):
                peers.append(table)
        if peers:
            parts = sorted([owner,*peers], key=lambda t:t['bbox'][0])
            owner['part_element_ids'] = [t['element_id'] for t in parts]
            owner['img_paths'] = [t['img_path'] for t in parts if t.get('img_path')]
            owner['table_body'] = '\n'.join(t['table_body'] for t in parts)
            footnotes = []
            for part in parts:
                notes = part['footnote']
                for note in ([notes] if isinstance(notes,str) else notes):
                    if note not in footnotes:
                        footnotes.append(note)
            owner['footnote'] = footnotes
            owner['bbox'] = [min(t['bbox'][0] for t in parts), min(t['bbox'][1] for t in parts),
                             max(t['bbox'][2] for t in parts), max(t['bbox'][3] for t in parts)]
            first = min(result['elements'].index(t) for t in parts)
            result['elements'] = [e for e in result['elements'] if all(e is not t for t in parts)]
            result['elements'].insert(first,owner)
            result['tables'] = [e for e in result['elements'] if e['type']=='table']


def _caption_positions(content_file):
    middle = content_file.with_name(content_file.name.replace('_content_list.json', '_middle.json'))
    if not middle.is_file():
        return []
    data = json.loads(middle.read_text(encoding='utf-8'))
    positions = []

    def walk(block, page, width, height):
        if block.get('type') in ('table_caption', 'image_caption') and block.get('bbox'):
            text = ' '.join(span.get('content', '') for line in block.get('lines', [])
                            for span in line.get('spans', []))
            box = block['bbox']
            positions.append({'text': text, 'page': page,
                'bbox': [box[0]/width*1000, box[1]/height*1000,
                         box[2]/width*1000, box[3]/height*1000]})
        for child in block.get('blocks', []):
            walk(child, page, width, height)

    for page in data.get('pdf_info', []):
        width, height = page['page_size']
        for block in page.get('para_blocks', []):
            walk(block, page['page_idx']+1, width, height)
    return positions


def _heading_level(text: str, reported: int) -> int:
    # The pipeline often labels every title level=1. Explicit numbering carries
    # the actual hierarchy, e.g. 4 -> 4.1 -> 4.1.2 and A -> A.1.
    numbered = re.match(r"^\s*((?:\d+|[A-Z])(?:\.\d+)*)(?:[.)、]?\s+)", text)
    if numbered:
        return numbered.group(1).count('.') + 1
    if re.match(r"^第[一二三四五六七八九十百零〇\d]+节", text):
        return 2
    return reported


def normalize_content(content: list[dict], caption_positions=None) -> dict:
    """Keep reading order and associate captions/footnotes with their owner."""
    result = {"normalization_version": NORMALIZATION_VERSION,
              "body": [], "tables": [], "images": [], "elements": []}
    headings = []
    for index, item in enumerate(content):
        kind = item.get("type")
        common = {"element_id": index, "page": int(item.get("page_idx", 0)) + 1,
                  "bbox": item.get("bbox")}
        if kind == "text" and item.get("text_level", 0) and not _caption_key(item.get('text', '')):
            level = _heading_level(item.get("text", ""), int(item["text_level"]))
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, item.get("text", "").strip()))
        common["section"] = " / ".join(title for _, title in headings)
        common["section_path"] = [title for _, title in headings]
        if kind in ("text", "equation", "list"):
            text = item.get("text", "")
            if kind == "list":
                text = "\n".join(item.get("list_items", []))
            element = {**common, "type": "text", "text": text.strip(),
                       "text_level": item.get("text_level", 0), "original_type": kind}
            result["body"].append(element)
        elif kind in ("table", "image"):
            prefix = "table" if kind == "table" else "image"
            element = {**common, "type": kind, "img_path": item.get("img_path", ""),
                       "caption": item.get(f"{prefix}_caption", []),
                       "footnote": item.get(f"{prefix}_footnote", [])}
            if kind == "table":
                element["table_body"] = item.get("table_body", "")
            result["tables" if kind == "table" else "images"].append(element)
        else:
            continue  # Discard running headers/footers/page numbers.
        result["elements"].append(element)
    _associate_captions(result, caption_positions or [])
    return result


def _extract_result(content_file: Path, work: Path, pdf_path: Path) -> dict:
    try:
        content = json.loads(content_file.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            raise ValueError("Expected a content list")
        result = normalize_content(content, _caption_positions(content_file))
    except (ValueError, TypeError, AttributeError) as exc:
        raise MinerUError("MinerU 结构化结果格式无效，请使用 requirements-mineru.txt 指定的版本。") from exc
    recover_table_symbols(pdf_path, result['tables'])
    for element in result["elements"]:
        if element.get("img_path"):
            asset = content_file.parent / element["img_path"]
            element["img_path"] = asset.relative_to(work).as_posix()
        if element.get('img_paths'):
            element['img_paths'] = [(content_file.parent / path).relative_to(work).as_posix()
                                    for path in element['img_paths']]
    for name in ("body", "tables", "images"):
        (work / f"{name}.json").write_text(
            json.dumps(result[name], ensure_ascii=False, indent=2), encoding="utf-8")
    (work / "parsed.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _settings() -> dict:
    return {"backend": "pipeline", "method": "auto", "table": True,
            "lang": os.getenv("MINERU_LANG", "ch"),
            "device": os.getenv("MINERU_DEVICE", "cpu"),
            "source": os.getenv("MINERU_MODEL_SOURCE", "huggingface")}


def _executable() -> str:
    configured = os.getenv("MINERU_EXECUTABLE")
    if configured:
        return configured
    local = Path(sys.executable).parent / ("mineru.exe" if os.name == "nt" else "mineru")
    if local.is_file():
        return str(local)
    found = shutil.which("mineru")
    if found:
        return found
    raise MinerUUnavailable(
        "未找到 MinerU。请在独立环境安装 requirements-mineru.txt，"
        "并将 MINERU_EXECUTABLE 设置为该环境的 mineru 可执行文件路径。")


def _cache_target(pdf_path: Path) -> Path:
    pdf_path = Path(pdf_path).resolve()
    settings = _settings()
    digest = hashlib.sha256(pdf_path.read_bytes())
    digest.update(json.dumps({"schema": 1, "mineru_version": "2.5.4", **settings}, sort_keys=True).encode())
    return pdf_path.parent / '.mineru' / digest.hexdigest()


def read_cached_pdf(pdf_path: Path) -> dict | None:
    """Read available preview elements without running OCR or changing an index."""
    target = _cache_target(pdf_path)
    try:
        result = json.loads((target/'parsed.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    return {**result, 'artifact_dir': str(target)}


def parse_pdf(pdf_path: Path) -> dict:
    """Cache by PDF bytes/settings and retain image files alongside the JSON."""
    pdf_path = Path(pdf_path).resolve()
    settings = _settings()
    target = _cache_target(pdf_path)
    root = target.parent
    with _PARSE_LOCK:
        if (target / "parsed.json").is_file():
            result = json.loads((target / "parsed.json").read_text(encoding="utf-8"))
            if result.get("normalization_version") != NORMALIZATION_VERSION:
                files = list((target / "raw").rglob("*_content_list.json"))
                if len(files) != 1:
                    raise MinerUError("MinerU 缓存缺少原始结构化结果，请移除该文档的缓存后重新解析。")
                result = _extract_result(files[0], target, pdf_path)
            return {**result, "artifact_dir": str(target)}
        executable = _executable()
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="parse-", dir=root) as temporary:
            work = Path(temporary) / "result"
            work.mkdir()
            command = [executable, "-p", str(pdf_path), "-o", str(work / "raw"),
                       "-b", settings["backend"], "-m", settings["method"],
                       "-l", settings["lang"], "-d", settings["device"],
                       "--source", settings["source"], "-t", "true"]
            try:
                # Stream diagnostics to disk so long CPU parses can be inspected
                # while running; keep the successful log with extraction assets.
                with (work / "mineru.log").open("w", encoding="utf-8") as output:
                    completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                               text=True, encoding="utf-8", errors="replace",
                                               timeout=int(os.getenv("MINERU_TIMEOUT", "1800")),
                                               check=False)
            except OSError as exc:
                raise MinerUUnavailable(f"无法启动 MinerU：{exc}") from exc
            except subprocess.TimeoutExpired as exc:
                raise MinerUError("MinerU 解析超时，请检查模型下载、设备配置或调大 MINERU_TIMEOUT。") from exc
            if completed.returncode:
                details = (work / "mineru.log").read_text(encoding="utf-8", errors="replace")[-2000:]
                raise MinerUError(f"MinerU 解析失败，请检查模型和 PDF。{details}")
            files = list((work / "raw").rglob("*_content_list.json"))
            if len(files) != 1:
                raise MinerUError("MinerU 未生成唯一的 *_content_list.json 结构化结果。")
            result = _extract_result(files[0], work, pdf_path)
            # Publish a complete directory atomically, never a partial cache.
            try:
                work.rename(target)
            except OSError:
                if not (target / "parsed.json").is_file():
                    raise
                # Another API process may have finished the same PDF first.
                result = json.loads((target / "parsed.json").read_text(encoding="utf-8"))
        return {**result, "artifact_dir": str(target)}
