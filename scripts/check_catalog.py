"""Check resource metadata and local Markdown file references (no network)."""
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / 'resources/catalog.json').read_text(encoding='utf-8'))
    errors = []
    ids, urls = set(), set()
    required = {'id', 'title', 'competition', 'type', 'url', 'priority', 'value',
                'limitation', 'next_action', 'verification', 'checked_at'}
    for item in data['resources']:
        missing = required - item.keys()
        if missing:
            errors.append(f"Missing fields: {missing}")
            continue
        if item['id'] in ids or item['url'] in urls:
            errors.append(f"Duplicate resource: {item['id']}")
        ids.add(item['id'])
        urls.add(item['url'])
        if urlsplit(item['url']).scheme not in ('https', 'http'):
            errors.append(f"Invalid URL: {item['id']}")
        if item['priority'] not in ('P0', 'P1', 'P2'):
            errors.append(f"Invalid priority: {item['id']}")
    for file in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', file.read_text(encoding='utf-8')):
            target = target.split('#')[0]
            if not target or urlsplit(target).scheme:
                continue
            if not (file.parent / unquote(target)).exists():
                errors.append(f"Broken local link: {file.relative_to(ROOT)} -> {target}")
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"OK: {len(ids)} resources; local Markdown links resolve. External availability is not tested.")


if __name__ == '__main__':
    main()
