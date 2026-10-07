"""Resumable parallel official PyPI wheel downloads with SHA-256 verification."""
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / '.wheels'
DEST.mkdir(exist_ok=True)


def metadata(name):
    for attempt in range(10):
        try:
            with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/json', timeout=90) as response:
                data = json.load(response)
            break
        except Exception:
            if attempt == 9:
                raise
            time.sleep(min(2**attempt, 15))
    choices = [x for x in data['urls'] if 'win_amd64.whl' in x['filename']
               and ('cp312' in x['filename'] or 'py3-none' in x['filename'])]
    if len(choices) != 1:
        raise RuntimeError(f'Ambiguous compatible wheel for {name}')
    return choices[0]


def piece(wheel, index, start, end):
    path = DEST / (wheel['filename'] + f'.part{index}')
    expected = end-start+1
    if path.exists() and path.stat().st_size == expected:
        return path
    for attempt in range(8):
        try:
            offset = path.stat().st_size if path.exists() else 0
            req = urllib.request.Request(wheel['url'], headers={'Range': f'bytes={start+offset}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError('Server did not honor byte range')
                with path.open('ab') as stream:
                    while data := response.read(65536):
                        stream.write(data)
            if path.stat().st_size != expected:
                raise RuntimeError('Incomplete chunk')
            print(f'{wheel["filename"]} chunk {index+1} complete', flush=True)
            return path
        except Exception as exc:
            print(f'Retry chunk {index}: {exc}', flush=True)
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f'Could not download {path.name}')


def main():
    wheels = [metadata(name) for name in ['xgboost', 'pyarrow', 'llvmlite']]
    pending = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for wheel in wheels:
            path = DEST/wheel['filename']
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == wheel['digests']['sha256']:
                continue
            size = 2*1024*1024
            for index, start in enumerate(range(0, wheel['size'], size)):
                pending.append(executor.submit(piece, wheel, index, start, min(start+size-1, wheel['size']-1)))
        for future in as_completed(pending):
            future.result()
    for wheel in wheels:
        path = DEST/wheel['filename']
        if not path.exists():
            with path.open('wb') as output:
                for i in range((wheel['size']+2*1024*1024-1)//(2*1024*1024)):
                    output.write((DEST/(wheel['filename']+f'.part{i}')).read_bytes())
        if hashlib.sha256(path.read_bytes()).hexdigest() != wheel['digests']['sha256']:
            raise RuntimeError(f'Checksum mismatch for {path.name}')
        print(f'VERIFIED {path.name}: {path.stat().st_size/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
