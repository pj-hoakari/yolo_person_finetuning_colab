"""Build an output-free standalone Colab notebook from reviewed source files."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inline_source(name):
    """Keep ordinary Python readable; resolve internal imports through prior cells."""
    source = (ROOT / 'src' / name).read_text()
    lines = source.splitlines(True)
    internal = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level:
            internal.update(range(node.lineno - 1, node.end_lineno))
    return ''.join(line for index, line in enumerate(lines) if index not in internal).strip()


def build():
    cells = []

    def md(text):
        cells.append({'cell_type': 'markdown', 'id': f'cell-{len(cells):02d}', 'metadata': {}, 'source': text.strip().splitlines(True)})

    def code(text):
        cells.append({'cell_type': 'code', 'id': f'cell-{len(cells):02d}', 'metadata': {}, 'source': text.strip().splitlines(True), 'execution_count': None, 'outputs': []})

    md('''# 1. CrowdHumanを用いたYOLO追加学習
vboxは可視人物、hboxは頭部、bothは両方を学習します。データは非公開Driveに保管し、Colabで処理します。
[データ利用条件](https://www.crowdhuman.org/download.html)（非商用の研究・教育用、画像再配布禁止）と[モデルのライセンス](https://www.ultralytics.com/license)、クラウド処理の可否を確認してください。
Colab内のデータはランタイム終了で消えます。必要な結果は保存し、ノートブック共有前に出力を削除してください。
実データでのGPU学習・アプリ接続は開発環境では未検証です。''')
    md('''## 2. Colab実行環境とGPUの確認
「ランタイム → ランタイムのタイプを変更」でGPUを選択します。GPUなしでは学習を開始できません。''')
    code('''import os, sys, subprocess, warnings
from pathlib import Path
if 'google.colab' not in sys.modules:
    try:
        import google.colab
    except ImportError:
        raise RuntimeError('Google Colabで実行してください') from None
BOOTSTRAP = Path('/content/crowdhuman')
if BOOTSTRAP.resolve() != BOOTSTRAP:
    raise RuntimeError('保存先が別の場所へリンクされています')
BOOTSTRAP.mkdir(parents=True, exist_ok=True)
os.environ.update(YOLO_CONFIG_DIR=str(BOOTSTRAP / 'config'), YOLO_OFFLINE='true',
                  YOLO_AUTOINSTALL='false', YOLO_VERBOSE='false', WANDB_MODE='disabled',
                  WANDB_DISABLED='true', COMET_MODE='DISABLED', HF_HUB_OFFLINE='1')
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True) if __import__('shutil').which('nvidia-smi') else None
if gpu is None or gpu.returncode:
    warnings.warn('GPUが見つかりません。GPUランタイムを選択してください。')
else:
    print(gpu.stdout.strip())''')
    md('''## 3. ライブラリのインストール
必要なライブラリを導入します。失敗時は`/content/crowdhuman/install.log`を確認してください。
依存ライブラリを変更したら、ランタイムを再起動して最初から実行します。''')
    requirements = (ROOT / 'requirements.txt').read_text()
    code(f'''REQUIREMENTS = {requirements!r}
requirements_path = BOOTSTRAP / 'requirements.txt'
requirements_path.write_text(REQUIREMENTS)
with open(BOOTSTRAP / 'install.log', 'w') as log:
    installed = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(requirements_path)], stdout=log, stderr=log)
if installed.returncode:
    raise RuntimeError('インストール失敗。ランタイム内のinstall.logを確認してください。')
print('インストール完了')''')
    md('''## 4. 学習設定
学習モード・epoch数・batch・Drive保存先を指定します。初回は`DEBUG_MODE=True`、`EPOCHS=1`で確認してください。
利用条件の確認後、`TERMS_REVIEWED=True`にします。本学習ではDEBUGをFalseに戻し、別の`RUN_NAME`を使います。
データ復元は`RESTORE_PREPARED_FROM_DRIVE=True`にし、保存時と同じモード・DEBUG設定で手順2〜5→12へ進みます。
中断した学習の再開は`RESUME=True`にします。データ・設定・依存バージョンは保存時と揃えてください。''')
    code('''TRAIN_MODE = 'vbox'
MODES_TO_RUN = [TRAIN_MODE]  # 比較例: ['vbox', 'hbox']; 3モード: ['vbox', 'hbox', 'both']
BASE_MODEL = 'yolo26n.pt'
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640
SEED = 42
DEBUG_MODE = False
DEBUG_LIMIT = 32  # 各splitの先頭N画像。すべてのモードで同じ画像。
RUN_NAME = 'experiment'
RESUME = False  # 中断から再開する場合はTrue。Driveの最新保存を自動復元。
SAVE_EVERY_EPOCH_TO_DRIVE = True
DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/crowdhuman/checkpoints'
DRIVE_PREPARED_DIR = '/content/drive/MyDrive/crowdhuman/prepared'
RESTORE_PREPARED_FROM_DRIVE = False  # 手順2〜5の後、12から復元・再開する場合True
ALLOW_CPU = False
TERMS_REVIEWED = False
DRIVE_MOUNT_POINT = '/content/drive'
DRIVE_DATASET_DIR = '/content/drive/MyDrive/crowdhuman'
DOWNLOAD_MISSING_TO_DRIVE = True
HF_DATASET_REPO = 'sshao0516/CrowdHuman'
HF_DATASET_REVISION = 'd97203da87e348ea69f7a7633a57c21a956120a6'
EXPORT_ONNX = True
WEBAPP_CONTRACT = None  # 既存アプリから採取したinput/outputs辞書。未指定では互換性を確定しません。
BASE_DIR = '/content/crowdhuman'
RAW_DIR = f'{BASE_DIR}/raw'
PROCESSED_DIR = f'{BASE_DIR}/processed'
RUNS_DIR = f'{BASE_DIR}/runs'
EXPORT_DIR = f'{BASE_DIR}/exports'
assert TRAIN_MODE in ('vbox', 'hbox', 'both')
assert MODES_TO_RUN and len(set(MODES_TO_RUN)) == len(MODES_TO_RUN)
assert all(mode in ('vbox', 'hbox', 'both') for mode in MODES_TO_RUN)''')
    md('''## 5. ディレクトリと非公開設定
保存先を`/content/crowdhuman/`に準備し、外部の実験追跡・同期を無効にします。''')
    ignore_rules = (ROOT / '.gitignore').read_text()
    code(inline_source('privacy.py') + f'\n\nGITIGNORE_CONTENT = {ignore_rules!r}\n' + '''BASE = initialize(Path(BASE_DIR))
with local_console(BASE / 'setup.log'):
    YOLO = private_yolo()
print('ランタイム内の保存先とプライバシー設定を準備しました')''')
    md('''## 6. Google Driveへ保管し、/contentへコピー
Driveをマウントし、不足するZIP4個・ODGT2個をHugging Faceから取得します。
既存ファイルはDriveの`raw/archives/`（ZIP）、`raw/annotations/`（ODGT）に置いてください。
Colabへコピーして整合性を確認します。失敗時はDriveの`logs/download.jsonl`、Colabの`logs/drive_copy.jsonl`を確認してください。''')
    code(inline_source('dataset.py') + '\n\n' + '''def mount_dataset_drive(mount_point):
    from google.colab import drive
    drive.mount(mount_point)

if not TERMS_REVIEWED:
    raise RuntimeError('利用条件とGoogle Drive・Colabでの保存・処理可否を確認し、TERMS_REVIEWED=Trueにしてください。')
mount_dataset_drive(DRIVE_MOUNT_POINT)
drive_dataset = Path(DRIVE_DATASET_DIR)
mount_root = Path(DRIVE_MOUNT_POINT).absolute()
if mount_root not in drive_dataset.absolute().parents:
    raise ValueError('DRIVE_DATASET_DIRはマウントしたGoogle Drive配下を指定してください。')
if DOWNLOAD_MISSING_TO_DRIVE:
    with local_download_storage(drive_dataset):
        for part in ('raw/archives', 'raw/annotations', 'logs'):
            runtime_path(drive_dataset / part).mkdir(parents=True, exist_ok=True)
        drive_log = download_huggingface(HF_DATASET_REPO, HF_DATASET_REVISION,
                                         progress=DownloadProgress(notebook=True))
        print('Driveへの保管完了。取得ログ:', drive_log)
copy_log = copy_drive_dataset(drive_dataset, progress=DownloadProgress(notebook=True))
print('Colabへのコピー・検証完了。ログ:', copy_log)''')
    md('''## 7. ZIPの安全な解凍
空き容量を確認し、画像を`raw/images/`へ展開します。正常な既存画像は再利用します。''')
    code('''print(extract_archives())''')
    md('''## 8. ODGTの解析と画像の検証
画像の存在・サイズ・破損と、train/val間のID重複を確認します。DEBUGでも全データを検証します。''')
    code('''image_sizes, split_counts = inspect_dataset()
print({'validated_images': split_counts})''')
    md('''## 9. YOLO形式への変換
選択したモードのラベルを生成します。ignore対象は除外するため、CrowdHuman公式評価とは異なります。
DEBUGから本学習へ切り替える際は、`processed/<mode>`を削除して再変換してください。''')
    code(inline_source('converter.py') + '\n\n' + '''yaml_paths = {}
for mode in MODES_TO_RUN:
    yaml_paths[mode] = convert_dataset(mode, image_sizes, DEBUG_LIMIT if DEBUG_MODE else None)
    report = __import__('json').loads((yaml_paths[mode].parent / 'conversion.json').read_text())
    print(mode, report['counts'])''')
    md('''## 10. 合成テストとアノテーションの検証
合成データで変換をテストし、ラベルのクラス・座標・画像との対応を確認します。
失敗時は`synthetic_tests.log`を確認してください。''')
    code((ROOT / 'tools/notebook_tests.py').read_text() + '\n\n' + '''suite = unittest.defaultTestLoader.loadTestsFromTestCase(NotebookSyntheticTests)
with local_console(BASE / 'synthetic_tests.log'):
    result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise RuntimeError('合成テスト失敗。synthetic_tests.logを確認してください。')
print({'synthetic_tests': result.testsRun, 'status': 'passed'})
for mode in MODES_TO_RUN:
    print(mode, validate_labels(yaml_paths[mode].parent, mode))''')
    md('''## 11. data.yamlの確認とGoogle Driveへの進捗保存
データ設定を確認し、展開画像・ラベルを`DRIVE_PREPARED_DIR`へ保存します。完了表示を待ってください。
次回は`RESTORE_PREPARED_FROM_DRIVE=True`にして、手順2〜5→12で復元できます。''')
    code(inline_source('preparation.py') + '\n\n' + '''for mode in MODES_TO_RUN:
    report = validate_labels(yaml_paths[mode].parent, mode)
    print({'mode': mode, 'data_yaml_valid': True, 'counts': report})
if not TERMS_REVIEWED:
    raise RuntimeError('利用条件を確認してください。')
if Path(DRIVE_MOUNT_POINT).absolute() not in Path(DRIVE_PREPARED_DIR).absolute().parents:
    raise ValueError('DRIVE_PREPARED_DIRはマウントしたGoogle Drive配下に指定してください。')
prepared_settings = {'modes': MODES_TO_RUN, 'debug': DEBUG_MODE, 'debug_limit': DEBUG_LIMIT if DEBUG_MODE else None}
print('手順11までの進捗をDriveへ保存中です。完了までお待ちください。')
prepared_archive = save_prepared_data(DRIVE_PREPARED_DIR, prepared_settings)
print('手順11までのDrive保存完了:', str(prepared_archive))''')
    md('''## 12. YOLOモデルの読み込み
公式のYOLO26n重みを準備します。`BASE_MODEL`でYOLO26sも選択できます。
データ復元を有効にした場合は、手順11の保存を照合・復元します。Colab側のデータ保存先は空にしてください。''')
    code(inline_source('dataset.py') + '\n\n' + inline_source('converter.py') + '\n\n' + inline_source('preparation.py') + '\n\n' + '''if RESTORE_PREPARED_FROM_DRIVE:
    if not TERMS_REVIEWED:
        raise RuntimeError('利用条件を確認し、TERMS_REVIEWED=Trueにしてください。')
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_POINT)
    if Path(DRIVE_MOUNT_POINT).absolute() not in Path(DRIVE_PREPARED_DIR).absolute().parents:
        raise ValueError('DRIVE_PREPARED_DIRはマウントしたGoogle Drive配下に指定してください。')
    prepared_settings = {'modes': MODES_TO_RUN, 'debug': DEBUG_MODE, 'debug_limit': DEBUG_LIMIT if DEBUG_MODE else None}
    print('手順11の保存を照合・復元中です。完了までお待ちください。')
    yaml_paths = restore_prepared_data(DRIVE_PREPARED_DIR, prepared_settings)
    for mode in MODES_TO_RUN:
        validate_labels(yaml_paths[mode].parent, mode)
    print('手順11の保存を復元しました。12から続行します。')

''' + inline_source('checkpoints.py') + '\n\n' + inline_source('trainer.py') + '\n\n' + '''if SAVE_EVERY_EPOCH_TO_DRIVE and Path(DRIVE_MOUNT_POINT).absolute() not in Path(DRIVE_CHECKPOINT_DIR).absolute().parents:
    raise ValueError('DRIVE_CHECKPOINT_DIRはマウントしたGoogle Drive配下に指定してください。')
configs = [TrainConfig(mode=mode, model=BASE_MODEL, epochs=EPOCHS, batch=BATCH_SIZE,
                       imgsz=IMAGE_SIZE, seed=SEED, debug=DEBUG_MODE, run_name=RUN_NAME,
                       allow_cpu=ALLOW_CPU) for mode in MODES_TO_RUN]
for config in configs:
    config.validate()
if not RESUME:
    with local_console(BASE / 'model_setup.log'):
        base_model = load_model(configs[0])
        del base_model
print('モデル準備完了')''')
    md('''## 13. 追加学習とチェックポイント管理
選択したモードを学習し、結果を`runs/<mode>/<RUN_NAME>/`に保存します。詳細ログは`console.log`です。
既定では毎epochの重み・設定・CSVをDriveへ保存し、保存失敗時は停止します。
中断後は同じ条件で`RESUME=True`にすると、最後に保存したepochの次から再開します。
GPUメモリ不足時はbatchを下げ、別の`RUN_NAME`で開始してください。''')
    code('''checkpoint_display = None
def show_saved_epoch(saved):
    global checkpoint_display
    from html import escape
    from IPython.display import HTML, display
    content = HTML('<pre>Drive保存完了: epoch ' + str(saved['epoch']) + '<br>' + escape(saved['directory']) + '</pre>')
    if checkpoint_display is None:
        checkpoint_display = display(content, display_id=True)
    else:
        checkpoint_display.update(content)

trained_runs = {}
for config in configs:
    trained_runs[config.mode] = train(config, resume=RESUME,
                                      drive_runs_dir=DRIVE_CHECKPOINT_DIR if SAVE_EVERY_EPOCH_TO_DRIVE else None,
                                      checkpoint_status=show_saved_epoch)
    print(config.mode, '学習完了')''')
    md('''## 14. 評価
検証データでPrecision・Recall・mAP、学習時間、推論速度を測定します。
推論速度はモデルのみの値で、アプリ全体の処理時間は含みません。''')
    code(inline_source('evaluator.py') + '\n\n' + '''for config in configs:
    print(evaluate(config))''')
    md('''## 15. vbox・hboxの比較
評価結果を`exports/comparison.csv`にまとめます。`matched_conditions`で実験条件の一致を確認できます。
vboxとhboxは検出対象が異なるため、mAPだけで優劣を判断しないでください。''')
    code('''comparison_rows = comparison()
print({'comparison_csv': f'{EXPORT_DIR}/comparison.csv', 'rows': len(comparison_rows),
       'matched_conditions': all(row['matched_conditions'] for row in comparison_rows)})
for row in comparison_rows:
    if row['scope'] == 'all':
        print(row)''')
    md('''## 16. モデルのエクスポート
重み（PT・ONNX）と学習CSVを`exports/`へ出力します。
ONNXはopset 13・FP32で検査し、入出力仕様を`*_onnx_interface.json`へ保存します。
アプリとの互換性は別途確認してください。`WEBAPP_CONTRACT`を指定すると入出力仕様を照合できます。''')
    code('''exported_names = []
for config in configs:
    exported_names.extend(export_model(config, onnx_export=EXPORT_ONNX, contract=WEBAPP_CONTRACT))
print({'exported': exported_names})''')
    md('''## 17. 必要ファイルの明示ダウンロード
手順14で評価後、`DO_DOWNLOAD=True`にして実行すると、学習設定・精度・損失をTXTで取得できます。
ファイル名は`training_results_YYYYMMDD_HHMMSS_ffffff.txt`（日本時間）です。TXTのみなら手順16は不要です。
出力済みの重み・CSVとTXTのファイル名は自動設定します。TXTが不要なら`DOWNLOAD_RESULTS_TXT=False`にします。''')
    code('''DO_DOWNLOAD = False
DOWNLOAD_RESULTS_TXT = True
DOWNLOAD_NAMES = [
    name
    for config in configs
    for name in (f'{config.mode}_{config.run_dir.name}.pt',
                 f'{config.mode}_{config.run_dir.name}.onnx',
                 f'{config.mode}_{config.run_dir.name}_training.csv')
    if (Path(EXPORT_DIR) / name).is_file()
]
if (Path(EXPORT_DIR) / 'comparison.csv').is_file():
    DOWNLOAD_NAMES.append('comparison.csv')
if DO_DOWNLOAD:
    if DOWNLOAD_RESULTS_TXT:
        results_txt_name = export_results_txt(configs)
        DOWNLOAD_NAMES.append(results_txt_name)
    print('ダウンロード対象:', DOWNLOAD_NAMES)
    download_selected(DOWNLOAD_NAMES)
else:
    print('出力済みファイル:', DOWNLOAD_NAMES)
    print('DO_DOWNLOAD=TrueでTXTを生成してダウンロードします')''')
    md('''## 18. 一時データの削除（任意）
必要な結果を取得してから実行してください。`DELETE_DATA=True`でraw・processed、`DELETE_RUNS=True`で学習ログ・重みを削除します。
Driveとexportsは残ります。rawを残して再変換する場合は、`processed/<mode>`だけを手動で削除してください。''')
    code('''DELETE_DATA = False
DELETE_RUNS = False
import shutil
if DELETE_DATA:
    for part in ('raw', 'processed'):
        target = runtime_path(BASE / part)
        if target.exists():
            shutil.rmtree(target)
if DELETE_RUNS:
    target = runtime_path(BASE / 'runs')
    if target.exists():
        shutil.rmtree(target)
print('削除設定に従って処理しました')''')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'colab': {'name': 'crowdhuman_finetuning.ipynb', 'provenance': [], 'private_outputs': True},
                         'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                         'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'cells': cells}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'crowdhuman_finetuning.ipynb'
    rendered = json.dumps(build(), ensure_ascii=False, indent=2) + '\n'
    if args.check:
        if target.read_text() != rendered:
            raise SystemExit('Notebook is stale; run tools/build_notebook.py')
    else:
        target.write_text(rendered)


if __name__ == '__main__':
    main()
