"""Download the six original CrowdHuman files locally, into an ignored directory.

Run from any directory: python3 tools/download_crowdhuman.py
Only Python's standard library is needed. No data is uploaded or displayed.
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.privacy import local_download_storage, runtime_path
from src.dataset import HF_REPO_ID, HF_REVISION, download_huggingface, inventory, DownloadProgress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default=HF_REPO_ID)
    parser.add_argument('--revision', default=HF_REVISION)
    parser.add_argument('--quiet', action='store_true', help='disable live progress; diagnostic logs are still saved')
    args = parser.parse_args()
    root = PROJECT_ROOT / 'crowdhuman'
    ignored = subprocess.run(['git', '-C', str(PROJECT_ROOT), 'check-ignore', '-q', '--',
                              'crowdhuman/raw/download_source.json'])
    if ignored.returncode != 0:
        parser.error('crowdhuman/ must be ignored by Git before downloading real data')
    with local_download_storage(root):
        for relative in ('raw/archives', 'raw/annotations', 'raw/images', 'logs'):
            runtime_path(root / relative).mkdir(parents=True, exist_ok=True)
        print(f'Storage: {root}', flush=True)
        print(f'Source: {args.repo_id} @ {args.revision}', flush=True)
        print('Downloading archives and annotations only; no extraction or training.', flush=True)
        try:
            progress = None if args.quiet else DownloadProgress()
            log = download_huggingface(args.repo_id, args.revision, progress=progress)
            print('Final integrity verification:', flush=True)
            report = inventory(progress=progress)
        except (RuntimeError, ValueError, OSError) as error:
            print(str(error), file=sys.stderr, flush=True)
            return 1
        print(f'Validated dataset: {report}', flush=True)
        print(f'Download log: {log}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
