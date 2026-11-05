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
vbox=可視人物、hbox=頭部、both=person/headの2クラスです。モデルは参考実装のYOLO26nを使用します。
[公式配布・条件](https://www.crowdhuman.org/download.html) は非商用の研究・教育用途に限定し、画像再配布を禁止しています。
[Ultralyticsライセンス](https://www.ultralytics.com/license) も確認してください。

CrowdHumanの元ZIP・ODGTは非公開のGoogle Driveフォルダへ保存し、実行時に `/content/crowdhuman/` へコピーします。
展開・変換・学習・評価はColabの一時ストレージで行います。Driveフォルダの共有リンクを公開しないでください。
ColabはGoogleのクラウドです。利用条件でクラウド処理を認められない場合はこのノートブックを実行せず、ローカル用に別途移植してください。
ランタイム終了・リセットでColab上のコピーと重みが失われますが、Driveに保管した元ファイルは残ります。
画像・アノテーションの表示・公開・再配布は行いません。
Colabの「編集 → ノートブックの設定 → 保存時にコードセルの出力を除外する」を有効にしてください。
このノートブック自身にも出力除外メタデータを設定しています。共有前に全出力を消してください。

GPU学習・実データ・ブラウザアプリとの接続は開発環境では未検証です。''')
    md('''## 2. Colab実行環境とGPUの確認
Colabの「ランタイム → ランタイムのタイプを変更」でGPUを選択します。GPUがなければ警告し、学習時に停止します。''')
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
Ultralyticsを8.4.52に固定します。ColabのCUDA対応PyTorchを利用し、実際の依存バージョンは実験に記録します。
インストールログはランタイム内に保存します。依存を変更した場合はランタイムを再起動して最初から実行してください。''')
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
初回は `DEBUG_MODE=True`、少ないepochで動作確認してください。正式比較ではFalseに戻し、別のRUN_NAMEを使います。
`MODES_TO_RUN=["vbox", "hbox"]` で同じ設定の比較を一括実行できます。bothは2クラスです。
利用条件とクラウド処理の可否を確認後に `TERMS_REVIEWED=True` にします。認証情報は入力しません。
Driveの保管先をDRIVE_DATASET_DIRに指定します。初回はHugging Faceから不足ファイルだけをDriveへ保存し、次回から再利用します。
既に元ファイルをDriveへ配置した場合はDOWNLOAD_MISSING_TO_DRIVE=Falseにできます。HF_DATASET_REVISIONでデータの版を固定します。
手順11の保存から再開する場合は `RESTORE_PREPARED_FROM_DRIVE=True` にし、手順2〜5の後に12へ進みます。
保存時と同じMODES_TO_RUN・DEBUG_MODE・DEBUG_LIMITを指定してください。学習前ならRESUME=Falseのままです。
中断した同一実験だけ `RESUME=True` にします。設定・データ・依存バージョンが変わる再開は拒否します。''')
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
以下は通常のPython関数です。ノートブック内で直接確認・編集でき、srcファイルの取得・生成・importは不要です。
raw/archives・raw/images・raw/annotations・processed・runs・exportsを作成します。
外部追跡サービス・同期・analyticsのコールバックを無効にし、学習ログもこの配下だけに保存します。
GitHubで管理する開発用srcとこのノートブックは、tools/build_notebook.pyで同期できます。Colabでの実行にはこの.ipynbだけを使います。''')
    ignore_rules = (ROOT / '.gitignore').read_text()
    code(inline_source('privacy.py') + f'\n\nGITIGNORE_CONTENT = {ignore_rules!r}\n' + '''BASE = initialize(Path(BASE_DIR))
with local_console(BASE / 'setup.log'):
    YOLO = private_yolo()
print('ランタイム内の保存先とプライバシー設定を準備しました')''')
    md('''## 6. Google Driveへ保管し、/contentへコピー
Google Driveをマウントし、非公開の `MyDrive/crowdhuman/` に元の画像ZIP4個とODGT2個を保管します。
初回は[sshao0516/CrowdHuman](https://huggingface.co/datasets/sshao0516/CrowdHuman)から不足分だけをDriveへ取得します。
既にファイルがある場合は `raw/archives/` にZIP、`raw/annotations/` にODGTを配置してください。
次に6ファイルを `/content/crowdhuman/raw/` へコピーし、SHA-256・ZIP CRC・ODGT形式を確認します。
同じセッション内の正常なローカルコピーはDriveと一致すれば再利用します。内容の異なるローカルファイルは無言で上書きしません。
コピー・照合・検証の進捗は1つの表示欄に出力します。取得ログはDrive側の `logs/download.jsonl`、コピーログはColab側の `logs/drive_copy.jsonl` に保存します。
画像展開・ラベル変換・学習はこの後のセルでローカルコピーに対して実行します。共有・公開や外部追跡への送信は行いません。''')
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
raw/imagesに1組だけ展開します。想定外のパス・シンボリックリンク・重複画像を拒否します。
既存画像はCRCが一致すれば再展開しません。画像・アノテーションはすべてランタイム内に留まります。''')
    code('''print(extract_archives())''')
    md('''## 8. ODGTの解析と画像の検証
1行ずつJSONとして読み込み、全画像の存在・画像ヘッダの整合性・サイズを調べます。
同一split内の重複ID、train/val間のID重複はエラーにします。DEBUGでも入力全体の分離を確認します。''')
    code('''image_sizes, split_counts = inspect_dataset()
print({'validated_images': split_counts})''')
    md('''## 9. YOLO形式への変換
person以外を除外し、vboxはextra.ignore、hboxはhead_attr.ignoreを独立に判定します。
標準YOLOはignore領域を表現できません。除外領域の検出は背景に対する誤検出として扱われる可能性があり、CrowdHuman公式評価とは異なります。
bboxをクリッピング・正規化し、不正データ件数だけ記録します。画像はシンボリックリンクで共用します。
DEBUGと本番で同じprocessedを使い回すことは拒否します。切替時は下の手順に従いprocessedを明示的に削除してください。''')
    code(inline_source('converter.py') + '\n\n' + '''yaml_paths = {}
for mode in MODES_TO_RUN:
    yaml_paths[mode] = convert_dataset(mode, image_sizes, DEBUG_LIMIT if DEBUG_MODE else None)
    report = __import__('json').loads((yaml_paths[mode].parent / 'conversion.json').read_text())
    print(mode, report['counts'])''')
    md('''## 10. 合成テストとアノテーションの検証
合成画像・合成ODGTで変換処理をテストします。実画像の表示は行いません。
クラスID・正規化座標・画像とラベルの対応・split分離を検証します。空ラベル画像は背景画像として残します。''')
    code((ROOT / 'tools/notebook_tests.py').read_text() + '\n\n' + '''suite = unittest.defaultTestLoader.loadTestsFromTestCase(NotebookSyntheticTests)
with local_console(BASE / 'synthetic_tests.log'):
    result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise RuntimeError('合成テスト失敗。synthetic_tests.logを確認してください。')
print({'synthetic_tests': result.testsRun, 'status': 'passed'})
for mode in MODES_TO_RUN:
    print(mode, validate_labels(yaml_paths[mode].parent, mode))''')
    md('''## 11. data.yamlの確認とGoogle Driveへの進捗保存
変換時にdata.yamlを生成済みです。各モードにimages/train・images/val・labels/train・labels/valがあります。
vboxは0:person、hboxは0:head、bothは0:person / 1:headです。データの内容は表示しません。
検証後、展開画像・ODGT・変換済みラベルとYAMLをDRIVE_PREPARED_DIRへ非公開保存します。容量と転送時間が必要です。
保存完了の表示を確認してからランタイムを終了してください。
新しいランタイムではRESTORE_PREPARED_FROM_DRIVE=Trueにして手順2〜5を実行し、6〜11を飛ばして12へ進みます。''')
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
参考実装のyolo26n.pt（設定でyolo26s.ptも選択可能）を使用します。
公式Ultralytics重みをランタイムに取得し、同じセッションではキャッシュを再利用します。
外部からアップロードした信頼できない.ptは読み込みません。
RESTORE_PREPARED_FROM_DRIVE=Trueの場合、Driveをマウントし、手順11の保存を照合・復元します。
新しいランタイムでは先に手順2〜5を実行してください。手順6〜11の再実行は不要です。
復元先は空のローカルデータディレクトリに限ります。保存時と異なるモード・DEBUG設定や破損した保存は拒否します。''')
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
同じ初期モデル・seed・学習条件で各モードを学習します。結果はruns/<mode>/<RUN_NAME>です。
進捗や画像はノートブックに表示せず、ログはランタイムに保存します。T4でOOMが出たらbatchを8などに下げ、新しい実験名を使ってください。
毎epoch終了時にoptimizerを含む再開用last.ptとbest.pt・学習設定・CSVをGoogle Driveへ保存します。
保存先はDRIVE_CHECKPOINT_DIR/<mode>/<RUN_NAME>/epoch_XXXXXX_...です。SHA-256照合後に最新保存のポインタを更新します。
新しいランタイムでも、同じデータ・設定・RUN_NAMEでRESUME=Trueにして上から実行すると、Driveから復元して最後に保存できたepochの次から再開します。
再開するモードだけをMODES_TO_RUNに指定してください。目標epoch数まで完了したチェックポイントは再開対象になりません。
Drive保存に失敗した場合は学習を停止し、未保存のまま次のepochへ進みません。Drive保存中の強制終了では、その前に保存完了したepochから再開します。
本番比較はearly stoppingで実際のepoch数が異なる可能性もあるため、training CSVも確認してください。''')
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
各モデルをそれぞれのval正解ラベルで評価します。Precision・Recall・mAP50・mAP50-95、学習時間、推論ms/画像、重みMiBを記録します。
速度はbatch=1・FP32でのUltralytics validationのinference値です。前後処理、ブラウザ、追跡器を含むアプリ全体の速度ではありません。
学習時間は再開を含むセッション時間で、準備処理も含みます。''')
    code(inline_source('evaluator.py') + '\n\n' + '''for config in configs:
    print(evaluate(config))''')
    md('''## 15. vbox・hboxの比較
検出対象が異なるためmAPだけで優劣は判定できません。bothではクラス別の行も出力します。
複数モードを実行するか、設定のモードを切り替えて学習・評価を追加後、このセルを再実行してください。
matched_conditionsは保存した設定・データ指紋・依存・評価条件・GPU名の一致を表します。対象の同一性や完全な再現性を保証する値ではありません。
重みが変わった後の古い評価結果はエラーになります。''')
    code('''comparison_rows = comparison()
print({'comparison_csv': f'{EXPORT_DIR}/comparison.csv', 'rows': len(comparison_rows),
       'matched_conditions': all(row['matched_conditions'] for row in comparison_rows)})
for row in comparison_rows:
    if row['scope'] == 'all':
        print(row)''')
    md('''## 16. モデルのエクスポート
exportsに.pt・ONNX・学習CSVを出力します。画像・ラベル・run一式のZIPは作成しません。
ONNXはopset 13・dynamic・FP32、simplify=Falseで出力し、checkerと合成入力によるONNX Runtime CPU推論を実行します。
入出力名・shape・dtype・クラス定義は*_onnx_interface.jsonに記録します。YOLO26のend-to-end出力は、従来のYOLOの生出力を想定するアプリと異なる場合があります。
WEBAPP_CONTRACTに実アプリの入出力辞書を設定すると完全一致を検査します。未指定は互換性未検証です。
RGB/NCHW/0〜1、letterbox・座標逆変換、confidence・クラスID・NMS有無、ONNX Runtime Web対応opsetをアプリ側で検証してください。''')
    code('''exported_names = []
for config in configs:
    exported_names.extend(export_model(config, onnx_export=EXPORT_ONNX, contract=WEBAPP_CONTRACT))
print({'exported': exported_names})''')
    md('''## 17. 必要ファイルの明示ダウンロード
このセルは初期状態ではダウンロードしません。必要なファイル名をDOWNLOAD_NAMESに列挙し、DO_DOWNLOAD=Trueに変更して、このセルだけを明示的に実行してください。
.pt・.onnx・.csvに限定します。公開操作ではありません。学習済みモデルにもデータ由来の情報が含まれる可能性があるため、外部公開・商用利用前に条件を確認してください。''')
    code('''DO_DOWNLOAD = False
DOWNLOAD_NAMES = []  # 例: ['vbox_experiment.pt', 'comparison.csv']
if DO_DOWNLOAD:
    download_selected(DOWNLOAD_NAMES)
else:
    print('ダウンロード未実行')''')
    md('''## 18. 一時データの削除（任意）
学習済みモデルの取得を確認してから実行してください。初期状態では削除しません。
DELETE_DATA=Trueはrawとprocessedを削除します。DELETE_RUNS=TrueはColab内のチェックポイントと学習ログを削除します。Drive側の保存は残るため、RESUME=Trueで復元できます。
DEBUGから本番へ切り替える際は、DELETE_DATAを使うとrawも消えます。rawを再利用する場合はColabファイル欄でprocessed/<mode>だけを明示的に削除し、新しいRUN_NAMEで変換・学習してください。
exportsは残ります。ランタイムのリセットではexportsも含めて消失します。
このセルはDriveの保管フォルダを削除しません。次のランタイムではDriveから再コピーできます。''')
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
