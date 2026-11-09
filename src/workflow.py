"""Prepare or restore the selected training data through one notebook step."""
import json
from pathlib import Path
from .privacy import local_download_storage, runtime_path
from .dataset import download_huggingface, copy_drive_dataset, extract_archives, inspect_dataset
from .converter import convert_dataset, validate_labels
from .preparation import save_prepared_data, restore_prepared_data


def prepare_training_data(modes, debug, debug_limit, drive_dataset_dir, drive_prepared_dir,
                          hf_repo, hf_revision, restore=False, save_prepared=True,
                          download_missing=True, progress=None, status=print):
    settings = {'modes': list(modes), 'debug': debug, 'debug_limit': debug_limit if debug else None}
    if restore:
        status('Driveの準備済みデータを復元中です')
        paths = restore_prepared_data(drive_prepared_dir, settings)
    else:
        if download_missing:
            status('不足する元ファイルをDriveへ取得中です')
            with local_download_storage(drive_dataset_dir):
                for part in ('raw/archives', 'raw/annotations', 'logs'):
                    runtime_path(Path(drive_dataset_dir) / part).mkdir(parents=True, exist_ok=True)
                download_huggingface(hf_repo, hf_revision, progress=progress)
        status('DriveからColabへコピー・検証中です')
        copy_drive_dataset(drive_dataset_dir, progress=progress)
        status('ZIPを展開中です')
        extract_archives()
        status('画像とアノテーションを検証中です')
        sizes, counts = inspect_dataset()
        status({'validated_images': counts})
        paths = {mode: convert_dataset(mode, sizes, debug_limit if debug else None) for mode in modes}
    for mode, yaml_path in paths.items():
        counts = validate_labels(yaml_path.parent, mode)
        conversion = json.loads((yaml_path.parent / 'conversion.json').read_text())
        if conversion['limit'] != settings['debug_limit']:
            raise ValueError('Converted DEBUG limit differs; remove processed/<mode> and convert again')
        status({'mode': mode, 'labels': counts})
    if not restore and save_prepared:
        status('準備済みデータをDriveへ保存中です。完了までお待ちください')
        archive = save_prepared_data(drive_prepared_dir, settings)
        status('準備済みデータのDrive保存完了: ' + str(archive))
    status('データ準備完了')
    return paths
