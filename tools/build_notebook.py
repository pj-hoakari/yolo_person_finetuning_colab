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

    md('''# 1. CrowdHumanでYOLOを追加学習
手順2〜9を上から実行します。初回は手順2のDEBUG設定で確認してください。
参考: [YOLO追加学習ノートブック](https://github.com/KasumiMercury/yolo_person_finetuning_colab/blob/main/yolo_person_finetuning.ipynb)。

vboxは可視人物、hboxは頭部、bothは両方です。元データは非公開Drive、学習はColabのGPUで行います。
[データ利用条件](https://www.crowdhuman.org/download.html)と[モデルのライセンス](https://www.ultralytics.com/license)、クラウド処理の可否を確認してください。
Colab内のデータはランタイム終了で消えます。結果を取得し、共有前にノートブックの出力を削除してください。''')
    md('''## 2. 設定
GPUランタイムを選び、モード・学習回数・Drive保存先を設定します。利用条件の確認後、`TERMS_REVIEWED=True`にしてください。
初回は`DEBUG_MODE=True`、`EPOCHS=1`で確認し、本学習ではDEBUGをFalseに戻して実験名を変えます。
データ復元は`RESTORE_PREPARED_FROM_DRIVE=True`、学習再開は同じ条件・実験名で`RESUME=True`にして、手順3以降を実行します。''')
    code('''# 学習
MODES_TO_RUN = ['vbox']  # ['hbox'] / ['vbox', 'hbox'] / ['both']
BASE_MODEL = 'yolo26n.pt'  # yolo26s.ptも選択可
RUN_NAME = 'experiment'
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640
PATIENCE = 10
WORKERS = 2
SEED = 42
DEBUG_MODE = False
DEBUG_LIMIT = 32  # DEBUG時の各splitの画像数
RESUME = False
ALLOW_CPU = False  # 意図的なCPUテストのみTrue
TERMS_REVIEWED = False

# Driveとデータ
DRIVE_MOUNT_POINT = '/content/drive'
DRIVE_DATASET_DIR = '/content/drive/MyDrive/crowdhuman'
DRIVE_PREPARED_DIR = '/content/drive/MyDrive/crowdhuman/prepared'
DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/crowdhuman/checkpoints'
DOWNLOAD_MISSING_TO_DRIVE = True
SAVE_PREPARED_TO_DRIVE = True  # 展開・変換済みデータの保存（追加容量が必要）
RESTORE_PREPARED_FROM_DRIVE = False
SAVE_EVERY_EPOCH_TO_DRIVE = True
HF_DATASET_REPO = 'sshao0516/CrowdHuman'
HF_DATASET_REVISION = 'd97203da87e348ea69f7a7633a57c21a956120a6'

# 結果の表示・出力
SHOW_LEARNING_CURVES = True  # 数値のグラフのみ
EXPORT_MODELS = True  # TXTのみ必要ならFalse
EXPORT_ONNX = True
WEBAPP_CONTRACT = None
DO_DOWNLOAD = False  # 手順9でダウンロードする場合True
DOWNLOAD_RESULTS_TXT = True

BASE_DIR = '/content/crowdhuman'
EXPORT_DIR = f'{BASE_DIR}/exports'
assert MODES_TO_RUN and len(set(MODES_TO_RUN)) == len(MODES_TO_RUN)
assert all(mode in ('vbox', 'hbox', 'both') for mode in MODES_TO_RUN)
assert DEBUG_LIMIT > 0
print('設定完了:', MODES_TO_RUN, RUN_NAME)''')
    md('''## 3. 環境準備とDrive接続
ライブラリ・保存先を準備し、Driveを接続します。認証画面に従ってください。
GPUなしでは停止します。インストール失敗時は`/content/crowdhuman/install.log`を確認します。''')
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
    if not ALLOW_CPU:
        raise RuntimeError('GPUランタイムを選択して再実行してください。')
    warnings.warn('CPUテストとして実行します。')
else:
    print(gpu.stdout.strip())
if not TERMS_REVIEWED:
    raise RuntimeError('利用条件を確認し、手順2のTERMS_REVIEWED=Trueを設定してください。')''')
    requirements = (ROOT / 'requirements.txt').read_text()
    code(f'''REQUIREMENTS = {requirements!r}
requirements_path = BOOTSTRAP / 'requirements.txt'
requirements_path.write_text(REQUIREMENTS)
with open(BOOTSTRAP / 'install.log', 'w') as log:
    installed = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(requirements_path)], stdout=log, stderr=log)
if installed.returncode:
    raise RuntimeError('インストール失敗。ランタイム内のinstall.logを確認してください。')
print('インストール完了')''')
    ignore_rules = (ROOT / '.gitignore').read_text()
    code(inline_source('privacy.py') + f'\n\nGITIGNORE_CONTENT = {ignore_rules!r}\n' + '''BASE = initialize(Path(BASE_DIR))
with local_console(BASE / 'setup.log'):
    YOLO = private_yolo()
print('ランタイム内の保存先とプライバシー設定を準備しました')''' + '\n\n' + inline_source('dataset.py') + '\n\n' + inline_source('converter.py') + '\n\n' + inline_source('preparation.py') + '\n\n' + inline_source('checkpoints.py') + '\n\n' + inline_source('trainer.py') + '\n\n' + inline_source('evaluator.py') + '\n\n' + inline_source('workflow.py'))
    code('''configs = [TrainConfig(mode=mode, model=BASE_MODEL, epochs=EPOCHS, batch=BATCH_SIZE,
                       imgsz=IMAGE_SIZE, patience=PATIENCE, workers=WORKERS, seed=SEED,
                       debug=DEBUG_MODE, run_name=RUN_NAME, allow_cpu=ALLOW_CPU)
           for mode in MODES_TO_RUN]
for config in configs:
    config.validate()

def mount_dataset_drive(mount_point):
    from google.colab import drive
    drive.mount(mount_point)

mount_dataset_drive(DRIVE_MOUNT_POINT)
mount_root = Path(DRIVE_MOUNT_POINT).absolute()
for directory in (DRIVE_DATASET_DIR, DRIVE_PREPARED_DIR, DRIVE_CHECKPOINT_DIR):
    if mount_root not in Path(directory).absolute().parents:
        raise ValueError('Drive保存先はマウント先の配下に指定してください。')
print('環境準備・Drive接続完了')''')
    md('''## 4. データ準備
不足ファイルの取得、Colabへのコピー、展開、YOLO形式への変換をまとめて実行します。復元設定がTrueならDriveの準備済みデータを使います。
準備済みデータを保存する場合は完了表示を待ってください。DEBUGと本学習の切替時は`processed/<mode>`を削除して再変換します。
ignore対象は除外するため、評価方法はCrowdHuman公式と異なります。''')
    code((ROOT / 'tools/notebook_tests.py').read_text() + '\n\n' + '''suite = unittest.defaultTestLoader.loadTestsFromTestCase(NotebookSyntheticTests)
with local_console(BASE / 'synthetic_tests.log'):
    result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise RuntimeError('合成テスト失敗。synthetic_tests.logを確認してください。')
print('合成テスト完了:', result.testsRun)''')
    code('''yaml_paths = prepare_training_data(
    modes=MODES_TO_RUN, debug=DEBUG_MODE, debug_limit=DEBUG_LIMIT,
    drive_dataset_dir=DRIVE_DATASET_DIR, drive_prepared_dir=DRIVE_PREPARED_DIR,
    hf_repo=HF_DATASET_REPO, hf_revision=HF_DATASET_REVISION,
    restore=RESTORE_PREPARED_FROM_DRIVE, save_prepared=SAVE_PREPARED_TO_DRIVE,
    download_missing=DOWNLOAD_MISSING_TO_DRIVE,
    progress=DownloadProgress(notebook=True),
)''')
    md('''## 5. モデル準備
公式の事前学習重みを準備します。再開時は手順6で保存済みの重みを復元します。''')
    code('''for config in configs:
    validate_labels(yaml_paths[config.mode].parent, config.mode)
if not RESUME:
    with local_console(BASE / 'model_setup.log'):
        base_model = load_model(configs[0])
        del base_model
print('モデル準備完了:', BASE_MODEL)''')
    md('''## 6. 学習
選択したモードを学習します。ログと重みは`runs/<mode>/<RUN_NAME>/`に保存します。
既定で毎epochをDriveへ保存します。保存失敗時は停止し、中断後は最後の保存から再開できます。
GPUメモリ不足時はbatchを下げ、別の実験名で開始してください。''')
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
    md('''## 7. 学習結果と評価
実行epoch数・mAPが最良のepoch・損失と精度のグラフを表示し、最良の重みを検証データで評価します。
比較表は`exports/comparison.csv`に保存します。vboxとhboxは検出対象が異なるため、mAPだけで優劣を判断しないでください。''')
    code('''evaluation_results = []
for config in configs:
    summary, _ = training_history(config)
    print(config.mode, '学習サマリー:', summary)
    if SHOW_LEARNING_CURVES:
        plot_training_history(config)
    metrics = evaluate(config)
    evaluation_results.append(metrics)
    print(config.mode, '評価結果:', metrics)
comparison_rows = comparison(output=Path(EXPORT_DIR) / 'comparison.csv', configs=configs)
print('比較CSV:', str(Path(EXPORT_DIR) / 'comparison.csv'))
print('比較条件の一致:', all(row['matched_conditions'] for row in comparison_rows))''')
    md('''## 8. モデル出力（任意）
`EXPORT_MODELS=True`でPT・学習CSV、`EXPORT_ONNX=True`でONNXも出力します。TXTのみならスキップできます。
ONNXの入出力仕様はJSONに保存します。アプリとの互換性は別途確認してください。''')
    code('''exported_names = []
if EXPORT_MODELS:
    for config in configs:
        exported_names.extend(export_model(config, onnx_export=EXPORT_ONNX, contract=WEBAPP_CONTRACT))
    print('出力完了:', exported_names)
else:
    print('モデル出力をスキップしました')''')
    md('''## 9. 結果のダウンロード
`DO_DOWNLOAD=True`で実行すると、日時付きTXTと出力済みの重み・CSVを取得できます。ファイル名は自動設定します。
TXT名は`training_results_YYYYMMDD_HHMMSS_ffffff.txt`（日本時間）です。TXTが不要なら`DOWNLOAD_RESULTS_TXT=False`にします。''')
    code('''DOWNLOAD_NAMES = available_exports(configs, include_onnx=EXPORT_ONNX) if EXPORT_MODELS else []
if not EXPORT_MODELS and (Path(EXPORT_DIR) / 'comparison.csv').is_file():
    DOWNLOAD_NAMES.append('comparison.csv')
if DO_DOWNLOAD:
    if DOWNLOAD_RESULTS_TXT:
        DOWNLOAD_NAMES.append(export_results_txt(configs))
    print('ダウンロード対象:', DOWNLOAD_NAMES)
    download_selected(DOWNLOAD_NAMES)
else:
    print('出力済みファイル:', DOWNLOAD_NAMES)
    print('手順2をDO_DOWNLOAD=Trueにして再実行後、このセルを実行してください')''')
    md('''## 10. 一時データの削除（任意）
結果の取得後に実行します。`DELETE_DATA=True`でraw・processed、`DELETE_RUNS=True`でログ・重みを削除します。
Driveとexportsは残ります。rawを残して再変換する場合は`processed/<mode>`だけを手動で削除してください。''')
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
