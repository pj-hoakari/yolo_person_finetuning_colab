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

実データは `/content/crowdhuman/` だけに保存します。Google Driveはマウントしません。
ColabはGoogleのクラウドです。利用条件でクラウド処理を認められない場合はこのノートブックを実行せず、ローカル用に別途移植してください。
ランタイム終了・リセットでデータと重みが失われます。画像・アノテーションの表示、外部送信、出力保存は禁止します。
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
取得先はHugging Faceのsshao0516/CrowdHumanです。HF_DATASET_REVISIONに40桁のコミットSHAを指定してデータの版を固定します。
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
RESUME = False
ALLOW_CPU = False
TERMS_REVIEWED = False
AUTO_DOWNLOAD = True
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
    md('''## 6. Hugging Faceから取得またはアップロード
[sshao0516/CrowdHuman](https://huggingface.co/datasets/sshao0516/CrowdHuman)から、画像ZIP4個と元のODGT2個だけを取得します。
ノートブックの設定でリポジトリとコミットSHAを指定できます。test用画像や変換済みラベルは取得しません。
全6ファイルの完了数、処理中のファイル、ダウンロード／検証の進捗、速度と推定残り時間を1つの表示欄で更新します。
標準HTTPで公開ファイルをraw/archives・raw/annotationsへ直接保存します。Hugging FaceのSDK・外部キャッシュ・ログインは使用しません。
認証・アクセス制限・通信失敗があれば停止し、回避処理は行いません。その場合だけ、正規に取得した不足ファイルをアップロードします。
AUTO_DOWNLOAD=Falseは自動取得不能を既に確認した場合の手動経路です。
正常な既存ファイルは再取得しません。破損ファイルは確認のうえランタイム内から削除・交換してください。
失敗理由は画面に短く表示し、日時・失敗段階・対象ファイル・HTTPステータス・転送量・例外とスタック位置を `/content/crowdhuman/logs/download.jsonl` に追記します。
Colabのファイル欄からログを開けます。署名付きURL、認証情報、レスポンス本文やアノテーション内容は記録しません。
Hugging Faceからの取得は、画像・アノテーションのアップロードや再配布を許可するものではありません。''')
    code(inline_source('dataset.py') + '\n\n' + '''if not TERMS_REVIEWED:
    raise RuntimeError('公式利用条件とColabでの処理可否を確認し、TERMS_REVIEWED=Trueにしてください。')
if AUTO_DOWNLOAD:
    try:
        download_log = download_huggingface(HF_DATASET_REPO, HF_DATASET_REVISION, progress=DownloadProgress(notebook=True))
        print('必要ファイルを取得・再利用しました。ログ:', download_log)
    except RuntimeError as error:
        print(str(error))
        print('上記のログを確認してください。正規取得した不足ファイルは以下からアップロードできます。')
        upload_missing()
else:
    upload_missing()''')
    md('''## 7. ファイル整合性・空き容量確認
必須6ファイル、全ZIPのCRC、ODGT構造とCrowdHumanの件数(train 15,000 / val 4,370)を確認します。
データの内容や画像IDは表示せず件数だけを出力します。展開前に非圧縮サイズと空き容量も確認します。''')
    code('''print(inventory(progress=DownloadProgress(notebook=True)))''')
    md('''## 8. ZIPの安全な解凍
raw/imagesに1組だけ展開します。想定外のパス・シンボリックリンク・重複画像を拒否します。
既存画像はCRCが一致すれば再展開しません。画像・アノテーションはすべてランタイム内に留まります。''')
    code('''print(extract_archives())''')
    md('''## 9. ODGTの解析と画像の検証
1行ずつJSONとして読み込み、全画像の存在・画像ヘッダの整合性・サイズを調べます。
同一split内の重複ID、train/val間のID重複はエラーにします。DEBUGでも入力全体の分離を確認します。''')
    code('''image_sizes, split_counts = inspect_dataset()
print({'validated_images': split_counts})''')
    md('''## 10. YOLO形式への変換
person以外を除外し、vboxはextra.ignore、hboxはhead_attr.ignoreを独立に判定します。
標準YOLOはignore領域を表現できません。除外領域の検出は背景に対する誤検出として扱われる可能性があり、CrowdHuman公式評価とは異なります。
bboxをクリッピング・正規化し、不正データ件数だけ記録します。画像はシンボリックリンクで共用します。
DEBUGと本番で同じprocessedを使い回すことは拒否します。切替時は下の手順に従いprocessedを明示的に削除してください。''')
    code(inline_source('converter.py') + '\n\n' + '''yaml_paths = {}
for mode in MODES_TO_RUN:
    yaml_paths[mode] = convert_dataset(mode, image_sizes, DEBUG_LIMIT if DEBUG_MODE else None)
    report = __import__('json').loads((yaml_paths[mode].parent / 'conversion.json').read_text())
    print(mode, report['counts'])''')
    md('''## 11. 合成テストとアノテーションの検証
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
    md('''## 12. data.yamlの確認
変換時にdata.yamlを生成済みです。各モードにimages/train・images/val・labels/train・labels/valがあります。
vboxは0:person、hboxは0:head、bothは0:person / 1:headです。データの内容は表示しません。''')
    code('''for mode in MODES_TO_RUN:
    report = validate_labels(yaml_paths[mode].parent, mode)
    print({'mode': mode, 'data_yaml_valid': True, 'counts': report})''')
    md('''## 13. YOLOモデルの読み込み
参考実装のyolo26n.pt（設定でyolo26s.ptも選択可能）を使用します。
公式Ultralytics重みをランタイムに取得し、同じセッションではキャッシュを再利用します。
外部からアップロードした信頼できない.ptは読み込みません。''')
    code(inline_source('trainer.py') + '\n\n' + '''configs = [TrainConfig(mode=mode, model=BASE_MODEL, epochs=EPOCHS, batch=BATCH_SIZE,
                       imgsz=IMAGE_SIZE, seed=SEED, debug=DEBUG_MODE, run_name=RUN_NAME,
                       allow_cpu=ALLOW_CPU) for mode in MODES_TO_RUN]
for config in configs:
    config.validate()
if not RESUME:
    with local_console(BASE / 'model_setup.log'):
        base_model = load_model(configs[0])
        del base_model
print('モデル準備完了')''')
    md('''## 14. 追加学習とチェックポイント管理
同じ初期モデル・seed・学習条件で各モードを学習します。結果はruns/<mode>/<RUN_NAME>です。
進捗や画像はノートブックに表示せず、ログはランタイムに保存します。T4でOOMが出たらbatchを8などに下げ、新しい実験名を使ってください。
中断時のlast.ptから同じセッション内で再開できます。完了後にoptimizerが除去された重みはresumeできません。
本番比較はearly stoppingで実際のepoch数が異なる可能性もあるため、training CSVも確認してください。''')
    code('''trained_runs = {}
for config in configs:
    trained_runs[config.mode] = train(config, resume=RESUME)
    print(config.mode, '学習完了')''')
    md('''## 15. 評価
各モデルをそれぞれのval正解ラベルで評価します。Precision・Recall・mAP50・mAP50-95、学習時間、推論ms/画像、重みMiBを記録します。
速度はbatch=1・FP32でのUltralytics validationのinference値です。前後処理、ブラウザ、追跡器を含むアプリ全体の速度ではありません。
学習時間は再開を含むセッション時間で、準備処理も含みます。''')
    code(inline_source('evaluator.py') + '\n\n' + '''for config in configs:
    print(evaluate(config))''')
    md('''## 16. vbox・hboxの比較
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
    md('''## 17. モデルのエクスポート
exportsに.pt・ONNX・学習CSVを出力します。画像・ラベル・run一式のZIPは作成しません。
ONNXはopset 13・dynamic・FP32、simplify=Falseで出力し、checkerと合成入力によるONNX Runtime CPU推論を実行します。
入出力名・shape・dtype・クラス定義は*_onnx_interface.jsonに記録します。YOLO26のend-to-end出力は、従来のYOLOの生出力を想定するアプリと異なる場合があります。
WEBAPP_CONTRACTに実アプリの入出力辞書を設定すると完全一致を検査します。未指定は互換性未検証です。
RGB/NCHW/0〜1、letterbox・座標逆変換、confidence・クラスID・NMS有無、ONNX Runtime Web対応opsetをアプリ側で検証してください。''')
    code('''exported_names = []
for config in configs:
    exported_names.extend(export_model(config, onnx_export=EXPORT_ONNX, contract=WEBAPP_CONTRACT))
print({'exported': exported_names})''')
    md('''## 18. 必要ファイルの明示ダウンロード
このセルは初期状態ではダウンロードしません。必要なファイル名をDOWNLOAD_NAMESに列挙し、DO_DOWNLOAD=Trueに変更して、このセルだけを明示的に実行してください。
.pt・.onnx・.csvに限定します。公開操作ではありません。学習済みモデルにもデータ由来の情報が含まれる可能性があるため、外部公開・商用利用前に条件を確認してください。''')
    code('''DO_DOWNLOAD = False
DOWNLOAD_NAMES = []  # 例: ['vbox_experiment.pt', 'comparison.csv']
if DO_DOWNLOAD:
    download_selected(DOWNLOAD_NAMES)
else:
    print('ダウンロード未実行')''')
    md('''## 19. 一時データの削除（任意）
学習済みモデルの取得を確認してから実行してください。初期状態では削除しません。
DELETE_DATA=Trueはrawとprocessedを削除します。DELETE_RUNS=Trueはチェックポイントと学習ログも削除し、再開できなくなります。
DEBUGから本番へ切り替える際は、DELETE_DATAを使うとrawも消えます。rawを再利用する場合はColabファイル欄でprocessed/<mode>だけを明示的に削除し、新しいRUN_NAMEで変換・学習してください。
exportsは残ります。ランタイムのリセットではexportsも含めて消失します。''')
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
