# CrowdHumanでYOLOを追加学習

CrowdHumanをYOLO形式に変換し、ColabのGPUで追加学習・評価します。元データと再開用の重みは非公開のGoogle Driveに保存し、学習はColabのローカルSSDで行います。

[Colabで開く](https://colab.research.google.com/github/pj-hoakari/yolo_person_finetuning_colab/blob/main/crowdhuman_finetuning.ipynb)

[参考ノートブック](https://github.com/KasumiMercury/yolo_person_finetuning_colab/blob/main/yolo_person_finetuning.ipynb)の「環境準備 → データ準備 → 学習 → 結果確認 → 出力」に沿って、全10手順に再構成しました。CrowdHumanの変換、Driveからの復元、日時付きTXTとダウンロード名の自動設定に対応します。ノートブック単体で実行でき、`src/`の取得は不要です。

## 実行手順

ColabでGPUランタイムを選び、手順2〜9を上から実行してください。説明ごとに複数のコードセルがある場合は、そのすべてを順に実行します。

| 手順 | 内容 |
|---|---|
| 1 | 概要・利用条件 |
| 2 | 学習・Drive・出力の設定 |
| 3 | GPU確認、ライブラリ導入、Drive接続 |
| 4 | データ取得・展開・変換、またはDriveから復元 |
| 5 | 公式事前学習モデルの準備 |
| 6 | 学習と毎epochのDrive保存 |
| 7 | 学習曲線・epochサマリー・評価・比較CSV |
| 8 | PT・ONNX・学習CSVの出力（任意） |
| 9 | 日時付きTXTと出力済みファイルのダウンロード |
| 10 | Colab内の一時データ削除（任意） |

設定は手順2にまとめています。初回は`DEBUG_MODE=True`、`EPOCHS=1`で確認してください。DEBUGでも元ファイル全体の取得・検証が必要です。本学習ではDEBUGをFalseに戻し、`RUN_NAME`を変えます。既存の`processed/<mode>`は明示的に削除してから再変換してください。

学習モードは`MODES_TO_RUN`で選びます。

| 設定 | 検出対象 |
|---|---|
| `['vbox']` | 可視人物（0: person） |
| `['hbox']` | 頭部（0: head） |
| `['both']` | 可視人物と頭部（0: person / 1: head） |
| `['vbox', 'hbox']` | 2種類のモデルを順番に学習・比較 |

既定はYOLO26n、50 epoch、batch 16、画像640です。GPUメモリ不足時はbatchを下げ、別の実験名で開始してください。

## データの準備と復元

手順2で`DRIVE_DATASET_DIR`を指定します。初回は手順4で不足分だけを[Hugging Face](https://huggingface.co/datasets/sshao0516/CrowdHuman)から取得します。取得元はコミット`d97203da87e348ea69f7a7633a57c21a956120a6`に固定しています。

既に取得済みなら、以下に配置して`DOWNLOAD_MISSING_TO_DRIVE=False`にします。

```text
MyDrive/crowdhuman/
├── raw/archives/
│   ├── CrowdHuman_train01.zip
│   ├── CrowdHuman_train02.zip
│   ├── CrowdHuman_train03.zip
│   └── CrowdHuman_val.zip
└── raw/annotations/
    ├── annotation_train.odgt
    └── annotation_val.odgt
```

Driveから`/content/crowdhuman/`へコピーし、SHA-256・ZIP CRC・ODGT形式を確認します。展開時にも空き容量を確認します。内容の異なる既存ファイルは上書きせず停止します。

手順4の最後に、既定で展開・変換済みデータを`DRIVE_PREPARED_DIR`へ保存します。元ZIPとは別に容量と転送時間が必要です。保存不要なら`SAVE_PREPARED_TO_DRIVE=False`にします。保存を繰り返すとDrive使用量が増えます。

新しいランタイムでは、保存時と同じモード・DEBUG設定で`RESTORE_PREPARED_FROM_DRIVE=True`にし、手順2以降を実行します。手順4が復元に切り替わるため、取得・展開・変換は行いません。Colab側の復元先は空である必要があります。旧ノートブックの手順11で作成した準備済みデータも、設定が一致すれば復元できます。

## 学習の中断と再開

既定で毎epochの重み・学習設定・CSVを`DRIVE_CHECKPOINT_DIR/<mode>/<RUN_NAME>/epoch_.../`へ保存します。SHA-256照合後に`latest.json`を更新し、保存失敗時は学習を停止します。

再開時は同じデータ・モード・実験名・epoch数・依存バージョンで`RESUME=True`にします。手順6でDriveの最新保存を復元し、最後に保存したepochの次から再開します。学習前のデータ復元だけなら`RESUME=False`のままです。

完了済みの重みや、設定が異なるチェックポイントからの再開は拒否します。通常終了時のColab側`last.pt`はoptimizerが除去される場合があるため、再開にはDriveの保存を使います。

## 結果の確認とダウンロード

手順7で実行epoch数・最良epoch・学習曲線と、最良の重みのPrecision・Recall・mAP50・mAP50-95を確認できます。曲線が不要なら`SHOW_LEARNING_CURVES=False`にします。

最良epochは学習CSVのmAP50-95が最大のepochです。Ultralyticsが保存した`best.pt`の評価結果は別途表示します。推論速度はモデル部分の値で、アプリ全体の処理時間ではありません。vboxとhboxは検出対象が異なり、mAPだけで優劣は決められません。

| 結果 | Colab内の保存先 |
|---|---|
| epochごとの損失・精度 | `runs/<mode>/<RUN_NAME>/results.csv` |
| 学習ログ | 同フォルダの`console.log` |
| 評価結果 | 同フォルダの`evaluation.json` |
| 最良の重み | 同フォルダの`weights/best.pt` |
| 比較表 | `exports/comparison.csv` |
| 出力済みモデル | `exports/<mode>_<RUN_NAME>.pt` / `.onnx` |
| 学習結果TXT | `exports/training_results_YYYYMMDD_HHMMSS_ffffff.txt` |

すべて`/content/crowdhuman/`配下です。DEBUG時は実験名に`_debug`が付きます。

手順2を`DO_DOWNLOAD=True`にして再実行し、手順9を実行すると、選択中の実験の出力済み重み・CSVとTXTを取得できます。ファイル名の入力は不要です。TXTには学習設定・epochサマリー・評価・各epochの損失と精度を記録し、ファイル名に実行時の日本時間を付けます。

TXTのみ必要なら`EXPORT_MODELS=False`として手順8を省略できます。TXTが不要なら`DOWNLOAD_RESULTS_TXT=False`にします。PTだけなら`EXPORT_ONNX=False`にします。ONNXはopset 13・dynamic・FP32で出力し、checkerと合成入力のRuntime推論を実行します。入出力仕様は`*_onnx_interface.json`に記録します。既存アプリとの互換性は、前後処理を含めて別途確認してください。

**Colab内の結果はランタイム終了で消えます。** 毎epochのDrive保存には評価JSON・グラフ・exportsは含まれないため、必要な最終結果を手順9で取得してください。手順10はColab内だけを削除し、Driveは削除しません。

## 利用条件とデータの扱い

[CrowdHumanの配布条件](https://www.crowdhuman.org/download.html)は非商用の研究・教育用途に限定し、画像の再配布を禁止しています。アノテーションも非公開として扱います。[Ultralyticsのライセンス](https://www.ultralytics.com/license)とDrive・Colabでの処理可否を確認後、`TERMS_REVIEWED=True`にしてください。

実画像の表示・予測画像の保存・外部実験追跡は無効です。学習曲線は損失・精度の数値のみを描画します。元データ・重み・認証情報はGitへ追加せず、Driveフォルダを公開しないでください。公開前にはノートブックの出力を削除します。

参考実装のCOCO単一クラス設定は、CrowdHumanの3モードへ置き換えています。ignore領域はラベルから除外するため、標準YOLOの評価はCrowdHuman公式評価とは異なります。事前学習モデルをheadラベルで比較する処理は設けていません。

## 開発と検証

`src/`と`tools/notebook_tests.py`を編集し、ノートブックを再生成します。

```bash
python3 -m pip install 'Pillow>=10,<13' 'PyYAML>=6,<7'
python3 tools/build_notebook.py
python3 -m unittest discover -s tests -v
python3 tools/build_notebook.py --check
python3 -m compileall -q src tests tools
```

`src/workflow.py`はデータ準備・復元、`src/evaluator.py`は評価・サマリー・出力を担当します。ノートブックには各ソースを通常のPythonセルとして埋め込みます。GitHub Actionsでも合成テストと生成内容の一致を確認します。

torch・Ultralyticsを導入した環境では、合成画像とランダム重みによるCPU結合試験も実行できます。

```bash
python3 tools/smoke_training.py
python3 tools/smoke_training.py --onnx-check
python3 tools/smoke_training.py --drive-resume
```

GPU・実データ・実際のDrive接続・ブラウザアプリの互換性は、この開発環境では未検証です。合成試験は実精度の保証にはなりません。

ローカルへ元データだけ取得する場合は`python3 tools/download_crowdhuman.py`を実行します。保存先はGit除外済みの`crowdhuman/`で、展開・学習は行いません。ノートブックはColab用です。

取得失敗時は、Driveの`logs/download.jsonl`（ローカル取得は`crowdhuman/logs/download.jsonl`）、コピー失敗時はColabの`logs/drive_copy.jsonl`を確認してください。トークンや署名付きURLはログに記録しません。
