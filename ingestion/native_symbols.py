"""Recover symbol-bearing native PDF rows within MinerU-detected table regions.

MinerU remains the layout/table parser. Native rows supplement OCR cells when
check/cross glyphs are missed, with coordinates retained for inspection.
"""
from __future__ import annotations

import re


def _glyph(char):
    value = char.get_text()
    if 'bbding' in char.fontname.lower():
        match = re.fullmatch(r'\(cid:(\d+)\)', value)
        code = int(match[1]) if match else ord(value) if len(value) == 1 else -1
        # bbding's builtin encoding; its empty ToUnicode map otherwise renders
        # these characters as !/% or cid:33/cid:37, not as their visible glyphs.
        return {33: '✓', 37: '✗'}.get(code, value)
    return value


def _chars(node):
    from pdfminer.layout import LTChar
    if isinstance(node, LTChar):
        yield node
    elif hasattr(node, '__iter__'):
        for child in node:
            yield from _chars(child)


def _line_text(chars):
    parts, previous = [], None
    for char in sorted(chars, key=lambda c: c.x0):
        if previous is not None and char.x0 - previous.x1 > 1.5:
            parts.append(' ')
        parts.append(_glyph(char))
        previous = char
    return ''.join(parts).strip()


def recover_table_symbols(pdf_path, tables):
    selected = [table for table in tables if table.get('bbox')]
    if not selected:
        return
    from pdfminer.high_level import extract_pages
    pages = sorted({table['page']-1 for table in selected})
    for page_number, layout in zip(pages, extract_pages(str(pdf_path), page_numbers=pages)):
        all_chars = list(_chars(layout))
        for table in selected:
            if table['page'] != page_number+1:
                continue
            box = table['bbox']
            x0, x1 = box[0]/1000*layout.width, box[2]/1000*layout.width
            y0, y1 = (1-box[3]/1000)*layout.height, (1-box[1]/1000)*layout.height
            chars = [c for c in all_chars if x0 <= (c.x0+c.x1)/2 <= x1
                     and y0 <= (c.y0+c.y1)/2 <= y1]
            symbols = [c for c in chars if _glyph(c) in {'✓','✔','✗','✘','√','×'}]
            if not symbols:
                continue
            lines = []
            for char in sorted(chars, key=lambda c: (-(c.y0+c.y1)/2, c.x0)):
                center = (char.y0+char.y1)/2
                line = next((line for line in reversed(lines) if abs(line[0]-center) <= 3), None)
                if line is None:
                    lines.append((center, [char]))
                else:
                    line[1].append(char)
            table['native_text'] = '\n'.join(_line_text(line) for _, line in lines)
            # Keep the indexed correction compact: the HTML already contains
            # numeric/text cells. Repeat row prefixes and symbol columns only.
            table['native_symbol_rows'] = []
            for _, line in lines:
                markers = [c for c in line if _glyph(c) in {'✓','✔','✗','✘','√','×'}]
                if markers:
                    end = max(c.x1 for c in markers)
                    table['native_symbol_rows'].append(_line_text([c for c in line if c.x0 <= end]))
            table['recovered_symbols'] = []
            for char in symbols:
                center = (char.y0+char.y1)/2
                row = min(lines, key=lambda line: abs(line[0]-center))[1]
                table['recovered_symbols'].append({'symbol': _glyph(char), 'row': _line_text(row),
                    'bbox': [char.x0/layout.width*1000, (1-char.y1/layout.height)*1000,
                             char.x1/layout.width*1000, (1-char.y0/layout.height)*1000],
                    'font': char.fontname})
