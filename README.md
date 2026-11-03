# CrowdHumanを用いたYOLO追加学習

CrowdHumanの元ZIP・ODGTを非公開のGoogle Driveフォルダに保管し、Colabの `/content/crowdhuman/` へコピーしてから変換・追加学習・評価を行います。`crowdhuman_finetuning.ipynb` をColabで開き、設定を確認して上から実行します。全処理と合成テストは、ノートブック内の通常のPythonセルとして収めています。外部のsrcファイルやテストファイルの取得・生成は不要です。

[Colabで開く](https://colab.research.google.com/github/pj-hoakari/yolo_person_finetuning_colab/blob/main/crowdhuman_finetuning.ipynb)

## GitHubでの管理

成果物は、このリポジトリのルートに置いています。`crowdhuman_finetuning.ipynb`だけをColabへアップロードして実行できます。GitHubで実装を変更する場合は、読みやすい差分を確認できる`src/`と`tools/notebook_tests.py`を編集し、`python3 tools/build_notebook.py`でノートブックのコードセルへ反映してください。実行時のsrc importやPythonファイル展開はありません。

`.github/workflows/tests.yml`は、push/PR時に合成データだけで変換・ノートブック単体動作・生成内容の一致を検証します。CrowdHuman取得、モデル学習、Artifacts・Releasesへの保存は行いません。`requirements.txt`はColab用で、CIではPillowとPyYAMLのみを導入します。

## 最初に確認すること

- [CrowdHuman公式配布ページ・利用条件](https://www.crowdhuman.org/download.html)では、非商用の研究・教育目的に用途を限定し、画像の再配布を禁止しています。本実装ではアノテーションも非公開として扱います。
- ColabはGoogleのクラウド環境です。今回調査した配布ページだけから、あらゆる利用条件でクラウド処理が許されるとは判断していません。適用される条件を確認し、クラウド処理が禁止される場合はColabを使わずローカル環境用に別途移植してください。ノートブックはColab用です。明示的なローカル取得には、下記のダウンロード専用コマンドを使用してください。
- [Ultralyticsのライセンス](https://www.ultralytics.com/license)はAGPL-3.0とEnterpriseの選択肢を案内しています。モデル・ライブラリの条件を確認してください。商用・非公開のWebアプリへの移行可否は、CrowdHumanの用途制限と独立に確認が必要です。
- 学習済みモデルにもデータ由来の情報が含まれる可能性があります。重みのダウンロードは公開の許可を意味しません。

## 参考実装との差分

[参考ノートブック](https://github.com/KasumiMercury/yolo_person_finetuning_colab/blob/main/yolo_person_finetuning.ipynb)を調査しました。

| 項目 | 参考実装 | 今回の実装 |
|---|---|---|
| モデル | yolo26n.pt（sへ変更可） | 同じモデルを優先 |
| データ | personに絞ったCOCO YOLO ZIP | 公式CrowdHuman ODGTを3モードへ変換 |
| 保存先 | DriveとColab SSD | 元ZIP・ODGTは非公開Drive、処理は`/content/crowdhuman/` |
| 初期設定 | 300 epoch / batch 32 / 640 | 50 epoch / batch 16 / 640 |
| 学習設定 | patience 10、cos_lr、close_mosaic、shear/perspective 0、AMP | 継承。cacheはFalse、workersは2、seed固定 |
| クラス | single_cls=True | bothを維持できるようFalse |
| 再開 | last.ptのepochとoptimizerを確認 | 継承し、設定・データ指紋・依存の一致も要求 |
| 可視化・保存 | グラフ表示、予測画像保存、Drive保存 | 画像表示なし、plot無効、スカラーCSVと重みだけ明示取得 |
| 出力 | CSV、重み、ONNX（opset 13） | 継承。入出力検査と比較CSVを追加 |

COCOのperson事前学習モデルをそのままheadの正解ラベルで比較する処理は設けていません。対象の異なるラベルとの比較に意味を持たせないためです。

## ファイル構成

```text
crowdhuman_finetuning.ipynb  # 出力なし・日本語19段階・コード同梱
src/dataset.py              # Hugging Face取得、ZIP/ODGT/画像検証、安全な展開
src/converter.py            # vbox / hbox / both変換、ラベル検査
src/trainer.py              # 学習設定、毎epoch保存、再開、実験情報
src/checkpoints.py          # Driveスナップショット、照合、ランタイム復元
src/evaluator.py            # 評価、比較CSV、ONNX検査、手動ダウンロード
src/privacy.py              # 保存先制限、外部連携停止、ログ制御
tests/test_converter.py     # 実データ不要の合成テスト
tools/build_notebook.py     # src/とノートブックの通常コードセルを同期
tools/notebook_tests.py     # ノートブックへ直接収める合成テスト
.github/workflows/tests.yml # 実データを使わないGitHub Actions
tools/smoke_training.py     # 合成画像とランダム重みによる任意CPU結合テスト
tools/download_crowdhuman.py # 実CrowdHumanの明示的なローカル取得
requirements.txt
.gitignore
```

## Colabの使い方

1. ノートブックをColabへアップロードし、GPUランタイムを選びます。「編集 → ノートブックの設定 → 保存時にコードセルの出力を除外する」も有効にします。共有前に全出力を削除してください。
2. セル2〜5で環境・依存・学習設定を準備します。`DRIVE_DATASET_DIR`にDrive保管先を指定します。Google Driveへの保存・Colabでの処理が利用条件に適合することを確認し、`TERMS_REVIEWED=True`にします。認証トークンはコードへ記述しません。
3. 最初は`DEBUG_MODE=True`、`EPOCHS=1`、必要に応じてbatchを下げて確認します。DEBUGは各splitの先頭`DEBUG_LIMIT`画像を使います。全モードで同じ画像を選びます。ダウンロード容量・入力全体の検証量は減りません。
4. セル6でDriveをマウントし、不足する元ファイルをHugging FaceからDriveへ保存して、`/content/crowdhuman/raw/`へコピーします。セル7〜12で検証・展開・変換・合成テストを実行します。画像IDやアノテーション自体は出力しません。
5. セル13〜17でモデル準備、学習、評価、比較、モデル出力を行います。GPUがない場合は警告して停止します。意図的なCPUテストだけ`ALLOW_CPU=True`を指定できます。
6. セル18で`DOWNLOAD_NAMES`に必要なファイルを列挙し、`DO_DOWNLOAD=True`としてそのセルを明示実行します。
7. 必要な重みの取得を確認後、セル19の削除フラグを必要なものだけ有効にします。

**ランタイム終了・リセットで、Colab内のコピー、展開画像、ラベル、ログ、チェックポイント、exportsは失われます。** Driveに保管した元ZIP・ODGTは残るため、次回はDriveから再コピーできます。手順12で展開・変換済みデータを保存し、手順13から復元できます。既定で毎epochの再開用チェックポイントもDriveへ保存します。配布・アプリ利用向けのPT/ONNX/CSVはセル17〜18で出力・取得してください。セル19の削除対象もColab側だけです。

### 手順12まで保存して13から再開

手順12の最後に、展開済み画像・ODGT・変換済みラベル・data.yaml・変換設定を `MyDrive/crowdhuman/prepared/` へ自動保存します。「手順12までのDrive保存完了」を確認してからランタイムを終了してください。元ZIPとは別に展開画像分のDrive容量と転送時間が必要です。再実行時は新しいスナップショットを残すため、使用容量が増えます。画像リンクはリンクとして保存し、画像の重複を避けます。

新しいランタイムでの操作:

1. 手順2〜5を実行します。手順4では保存時と同じ `MODES_TO_RUN`・`DEBUG_MODE`・`DEBUG_LIMIT` を指定し、`TERMS_REVIEWED=True`、`RESTORE_PREPARED_FROM_DRIVE=True` にします。まだ学習していなければ `RESUME=False` のままです。
2. 手順6〜12を飛ばして手順13を実行します。Driveのマウント後、SHA-256で保存を照合し、Colabへ復元してラベルを再検証します。ダウンロード・ZIP展開・変換の再実行は不要です。
3. 手順14以降を続けます。学習中の中断も復元する場合は、従来どおり同じ学習設定・RUN_NAMEで `RESUME=True` にします。

既に手順12まで完了したセッションでは、更新版の手順4の `DRIVE_PREPARED_DIR` と `DRIVE_MOUNT_POINT`、`RESTORE_PREPARED_FROM_DRIVE=False` を設定し、更新版の手順12を実行すれば保存できます。`RESTORE_PREPARED_FROM_DRIVE=False` は通常の上からの実行用です。同じランタイムで保存後そのまま13へ進む場合もFalseにします。

保存失敗時は停止し、最新保存の参照は前回の完了分を維持します。復元先は空のローカルデータディレクトリに限り、既存データがあれば停止します。Driveの実マウント・実データ転送は開発環境では未検証です。

### Google Driveへの保管とColabへのコピー

必要なファイルは次の6つです。

- `CrowdHuman_train01.zip` / `CrowdHuman_train02.zip` / `CrowdHuman_train03.zip`
- `CrowdHuman_val.zip`
- `annotation_train.odgt` / `annotation_val.odgt`

取得先は[sshao0516/CrowdHuman](https://huggingface.co/datasets/sshao0516/CrowdHuman)です。公開プロフィールでShuai Shao名義を確認し、元のZIP・ODGTが揃ったコミット`d97203da87e348ea69f7a7633a57c21a956120a6`に固定しています。Hugging FaceのカードにはCC BY-NC 4.0のメタデータと、CrowdHumanの非商用研究・教育／画像再配布禁止の条件が記載されています。取得先の変更によって画像・アノテーションを公開する権限が得られるわけではありません。

設定セルの`HF_DATASET_REPO`と`HF_DATASET_REVISION`で取得先を指定します。revisionには40桁のコミットSHAを指定します。変更する場合も、同じ6ファイルが揃い、利用が認められる配布元を使ってください。test用ZIPや変換済みの第三者ラベルは取得しません。vbox/hbox/ignoreは元のODGTから処理します。

設定の既定値は次のとおりです。

```python
DRIVE_MOUNT_POINT = '/content/drive'
DRIVE_DATASET_DIR = '/content/drive/MyDrive/crowdhuman'
DOWNLOAD_MISSING_TO_DRIVE = True
```

初回実行時はセル6でDriveの認証画面に従ってマウントします。Hugging Faceの公開resolve URLから必要な6ファイルをDriveへ直接取得します。Hugging FaceのSDKやトークンは使いません。認証・アクセス制限の回避処理もありません。正常な既存ファイルは再ダウンロードしません。

既にローカルへ取得したファイルがある場合は、Google DriveのWeb画面などから、次の非公開フォルダへ配置してください。共有リンクは公開しないでください。配置済みなら`DOWNLOAD_MISSING_TO_DRIVE=False`にするとネットワーク取得を行いません。

```text
MyDrive/crowdhuman/
├── raw/
│   ├── archives/
│   │   ├── CrowdHuman_train01.zip
│   │   ├── CrowdHuman_train02.zip
│   │   ├── CrowdHuman_train03.zip
│   │   └── CrowdHuman_val.zip
│   └── annotations/
│       ├── annotation_train.odgt
│       └── annotation_val.odgt
└── logs/download.jsonl
```

6ファイルが揃った後で、Colabの`/content/crowdhuman/raw/`へコピーします。転送量、SHA-256照合、ZIP CRC・ODGT検証の進捗を表示します。コピーは一時ファイルに行い、照合・検証に成功してから確定します。失敗時の不完全コピーは削除します。空き容量はコピーに必要な量と2 GiBの予約領域を要求し、展開前にも別途確認します。

同じColabセッションの再実行ではDriveとローカルのSHA-256が一致し、検証を通過したファイルを再利用します。異なるローカルファイルは上書きせず停止します。ログを確認して対象のColab側ファイルを明示削除し、再コピーしてください。Driveの元ファイルはコピー処理で変更・削除しません。

展開画像とYOLOラベルはColab内で処理し、手順12で非公開Driveへ進捗保存します。学習はColabで行い、再開用チェックポイント・実験設定・学習CSVを毎epoch Driveへ保存します。元ファイルをDriveに保管することは、画像・アノテーションの公開や再配布を許可するものではありません。

### 実CrowdHumanをローカルへダウンロードする

ノートブックはDrive保管・Colab処理が既定です。別途、ローカルで実ファイルを取得する場合は、リポジトリのルートで次を実行します。Python標準ライブラリだけで動作し、GPUや追加パッケージは不要です。取得前に前述のデータ利用条件を確認してください。

```bash
python3 tools/download_crowdhuman.py
```

保存先はリポジトリ直下の`crowdhuman/`です。Gitで除外されていることを確認してから開始します。ZIP4個とODGT2個を取得・検証し、展開・学習は行いません。正常な取得済みファイルは再利用できます。データのアップロード処理はありません。

ログは`crowdhuman/logs/download.jsonl`に追記します。取得不能なら終了コード1と理由を表示します。通常のターミナルから同じコマンドを再実行できます。ネットワークを制限された環境ではDNS/HTTPエラーになります。ローカル取得モードの保存先変更はそのコマンド実行中に限定され、Colabノートブックの既定保存先には影響しません。

### 進捗の表示

ローカル取得コマンドは標準で進捗を表示します。全6ファイルの準備完了数、現在のファイル、接続・転送・検証・再利用の状態、ファイル内の進捗率、処理量、平均速度、推定残り時間を確認できます。取得済みZIPのCRC検証中も、展開データをディスクへ保存せずストリームで検証量を更新します。ODGTは検証したレコード件数を表示します。表示例:

```text
[4/6 完了] ダウンロード中 annotation_train.odgt | 50.0% | 38.2/76.3 MiB | 10.0 MiB/s | 残り 00:03
[4/6 完了] 検証中 annotation_train.odgt | 66.7% | 10,000/15,000 件 | 5,000 件/s | 残り 00:01
[5/6 完了] 取得・検証完了 annotation_train.odgt
```

TTYでは同じ行を更新し、非TTYでは約1秒間隔と処理段階の変化時に出力します。総サイズが通知されない場合は割合と残り時間を`--`として受信量を表示します。速度と残り時間は現在の処理段階に対する推定値です。Colabでは1つの表示欄を更新します。表示を止めたい場合は`python3 tools/download_crowdhuman.py --quiet`を使います。診断ログは表示の有無に関係なく残ります。

### ダウンロード失敗時のログ

セル6で、Hugging Face取得のログはDriveの`MyDrive/crowdhuman/logs/download.jsonl`へ、DriveからColabへのコピーログは`/content/crowdhuman/logs/drive_copy.jsonl`へ保存します。エラー画面にもログの場所を表示します。Colabのファイル欄から開けます。失敗理由を記録するには、更新したセル6を再実行してください。過去の失敗は復元できません。ローカル取得CLIのログは引き続き`crowdhuman/logs/download.jsonl`です。

JSON Lines形式で、UTC日時・試行ID・処理段階・対象ファイル・HTTPステータス・受信先ホスト・転送量・想定サイズ・空き容量・例外の種類と理由・スタック位置を追記します。HTTP 403等は例外内の`http_status`、DNSエラーは`reason`、転送不足は`bytes_received`と`expected_bytes`、既存ファイル破損は`stage=existing_file_validation`を確認します。64 MiBごとの進捗、再利用、完了、中断も記録するため、同じセッションの再実行を試行IDで区別できます。

署名付きURL・認証情報・全レスポンスヘッダ・レスポンス本文・画像・アノテーション内容・ローカル変数は記録しません。スタックはファイル名、行番号、関数名だけです。ログの書き込み自体が失敗した場合は、その理由も画面に表示します。ログは指定した非公開DriveフォルダまたはColabの保存先に留め、Git管理や外部追跡への送信はしません。Colab側のログはランタイムのリセットで失われます。

### ランタイム内の構成

```text
/content/crowdhuman/
├── raw/
│   ├── archives/       # 4 ZIP
│   ├── images/         # 展開画像を1組だけ保持
│   └── annotations/    # 2 ODGT
├── processed/
│   ├── vbox/
│   ├── hbox/
│   └── both/           # 各モードにimages/{train,val}, labels/{train,val}, data.yaml
├── runs/{vbox,hbox,both}/<RUN_NAME>/
├── exports/
├── logs/               # ダウンロード診断JSONL（自動追記）
├── weights/            # 公式事前学習重み
└── config/             # 外部連携を無効にしたUltralytics設定

```

processedの画像はraw/imagesへのシンボリックリンクです。raw削除後は使えません。ZIPを保持したままの展開に必要な容量を事前に計算し、2 GiBの予約領域も要求します。ZIPのCRCを確認し、既存展開画像もCRC一致時だけ再利用します。画像はPillowのヘッダ検証を行いますが、すべての画像の全画素デコードまで事前保証するものではありません。

### モードとignoreの扱い

| モード | 正解bbox | クラス |
|---|---|---|
| vbox | 可視領域 | 0: person |
| hbox | 頭部 | 0: head |
| both | 可視領域と頭部 | 0: person、1: head |

fbox（全身領域）は使いません。`tag != person`は除外します。`extra.ignore=1`はvboxだけ、`head_attr.ignore=1`はhboxだけを除外します。省略された属性・ignoreは0として扱い、不正な属性形式は該当bboxを不正件数として記録します。両フラグをORして頭部と人物をまとめて消す処理は行いません。

bboxを画像境界へクリッピングしてからYOLOの中心x/中心y/幅/高さへ正規化します。ゼロ・負幅、非数、非有限値、クリッピング後の空矩形は除外します。モード別の`conversion.json`にsplit別の採用・ignore・不正・クリッピング件数を記録します。元データのカンマ入り画像IDを許可し、パス区切り・ドット・空白・制御文字を含むIDは拒否します。不正ODGT行や危険なID、画像不足、split重複、壊れたラベル/YAMLは停止条件です。

**標準YOLOラベルはignore領域を表現できません。** 除外された人物・頭部・maskの領域は学習時に背景と扱われ、評価時にもignore領域の検出が誤検出になり得ます。空ラベル画像も背景画像として残します。これはCrowdHuman公式のignore対応評価や公式指標の再現ではなく、変換後のYOLOラベルに対する評価です。画像を加工してignore領域を隠す処理は行っていません。

### 比較と再開

`MODES_TO_RUN=['vbox', 'hbox']`で条件を揃えて2モデルを順番に学習します。bothを追加する場合は`['vbox', 'hbox', 'both']`を使います。各モデルを同じ初期モデルから独立に初期化します。

結果は`exports/comparison.csv`です。Precision、Recall、mAP50、mAP50-95、学習セッション累計秒、推論ms/画像、重みMiBを保存します。bothはクラス別の行も追加します。`matched_conditions=False`なら学習設定・データ指紋・依存・評価条件・GPU名が揃っていません。vboxとhboxは検出対象が異なり、mAPの大小だけで優劣を決められません。1モードだけでもCSVを出せますが、比較には複数の実験が必要です。

推論速度はbatch=1・FP32のvalidation内部のinference時間です。前処理・後処理・ブラウザ・追跡は含みません。学習時間はモデル準備を含み、中断・再開分も合算します。early stoppingにより実際のepoch数が異なることがあります。各`results.csv`も確認してください。

毎epochの終了時に、optimizerを含む再開用チェックポイントをDriveへ保存します。設定の既定値は次のとおりです。

```python
SAVE_EVERY_EPOCH_TO_DRIVE = True
DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/crowdhuman/checkpoints'
RESUME = False
```

保存先は`checkpoints/<mode>/<RUN_NAME>/epoch_000001_<識別子>/`です。各スナップショットに`weights/last.pt`、存在する場合の`weights/best.pt`、`experiment.json`、`results.csv`、`args.yaml`を保存します。画像・ラベル・学習バッチは含めません。各ファイルのSHA-256を照合し、保存が完了してから`latest.json`を更新します。各epochの履歴を残すため、Drive使用量は保存回数に応じて増えます。

Colabのランタイムがリセットされた後も、次の手順で再開できます。

1. 元と同じ`RUN_NAME`、学習モード、EPOCHS等の学習設定、DEBUG設定を指定し、`RESUME=True`にします。`MODES_TO_RUN`には再開するモードだけを指定してください。
2. 手順12の保存があれば、上記の手順2〜5→13でデータを復元できます。保存がなければ上から実行してDriveをマウントし、元データをColabへコピー・再変換します。
3. 学習セルで、ローカルに実験がなければDriveの最新の保存済みスナップショットを自動復元します。実験設定・データ指紋・依存バージョン・ファイルSHA-256が一致することを確認してから、optimizer等を復元して次のepochから学習します。

例えばEPOCHS=50でepoch 12の保存完了後に停止した場合、設定を変えずに再開するとepoch 13〜50を学習します。途中のepochで切断された場合は、そのepochを最初からやり直します。目標epoch数を既に完了したチェックポイントは再開対象になりません。

Ultralyticsが学習終了時にoptimizerを削除する前のチェックポイントをDriveへ保存するため、Drive側には再開用状態が残ります。ローカルに完了後のoptimizer除去済みlast.ptがある場合は、それを直接resumeできません。元のDrive保存は残したまま該当するColab側のrunディレクトリを明示削除し、必要に応じてDriveから復元してください。ただし設定済みの最終epochに到達していれば再開を拒否します。

Drive保存に失敗すると学習を停止し、未保存のまま次のepochへ進みません。理由はColabのrunディレクトリ内`drive_backup_error.json`へ残します。保存途中の強制終了時は、`latest.json`が指す直前の保存完了スナップショットを使います。ノートブックには保存完了したepochとDriveの保存先を表示します。既存実験への新規学習の上書きは拒否します。

DEBUGから本番への移行やデータ変更時は、該当`processed/<mode>`をColabファイル欄から明示削除して再変換し、別のRUN_NAMEで新規学習します。rawは再利用できます。モードごとに実験を残し、比較CSVは保存済み条件が揃わない行も一致フラグをFalseにして表示します。

### 出力とWebアプリへの接続

`exports/<mode>_<RUN_NAME>.pt`、`.onnx`、`_training.csv`、`comparison.csv`がダウンロード対象です。DEBUGでは実験名に`_debug`が付きます。runディレクトリ一式をZIPにしないため、画像・ラベルの混入を避けられます。`files.download()`を呼ぶのはセル18でユーザーが明示指定した場合だけです。

ONNXはopset 13、dynamic、FP32、simplify=Falseで出力します。`onnx.checker`とONNX Runtime CPUの合成ゼロ画像で検査し、`*_onnx_interface.json`に入出力名・dtype・shape・クラスを記録します。既存アプリのコード・モデル仕様はこの作業場所に提供されていないため、ブラウザ互換性は未検証です。

`WEBAPP_CONTRACT`にはアプリが実際に要求する辞書を指定できます。生成レポートの`interface`と同じ構造（`input: {name, dtype, shape}`、`outputs: [{name, dtype, shape}]`）で完全一致を検査します。未指定なら`application_contract: unverified`です。一致してもブラウザでの結合試験の代替にはなりません。

アプリ側ではRGB、float32、NCHW、0〜1への正規化、letterbox、元画像への座標復元、confidence・class ID、ONNX Runtime Webのopset対応を確認してください。YOLO26のend-to-end出力は、従来のYOLOの生出力を前提とするデコーダと異なる可能性があります。実際の出力shapeと座標形式を確認し、NMSの有無と追跡器への入力も合わせて検証してください。

## プライバシーとGit

ノートブックは元ZIP・ODGTを指定した非公開Driveフォルダへ保存し、処理対象は`/content/crowdhuman/`配下のコピーに制限します。明示的なローカル取得CLIだけは、Git管理対象外のリポジトリ直下`crowdhuman/`を使用します。範囲外へのsymlinkも拒否します。Driveのマウントと元ファイルの保管・読み取りを行います。Hubアップロード、外部画像API、GitHub Releases/Artifacts/Actionsへの保存処理はありません。同期・外部実験追跡を設定で無効にし、HUB/Platform/analytics等のintegration callback登録も停止します。学習・検証のplot、画像表示、予測画像保存を使いません。大量ログはランタイム内のconsole.log等に保存し、ノートブックには要約だけを表示します。

これはOSのネットワーク遮断ではありません。パッケージ・公式重み・Hugging Face上のCrowdHumanの取得にはネットワークを使います。ユーザーが別途外部連携を有効にしないでください。ノートブックを共有する前に、出力除去とステージ済みファイルを確認してください。

`.gitignore`で指定のデータ・画像・重み・秘密情報・Pythonキャッシュを除外し、exportsも除外します。`git add -f`を防ぐ仕組みではないため、実データをリポジトリ内に配置しない運用を維持してください。認証情報のコード直書きはありません。開発用テストは独自作成の合成データだけを一時ディレクトリへ保存します。

## 開発と検証

```bash
# 変換・プライバシー・構造テストだけならGPU不要
python3 -m pip install Pillow PyYAML
python3 tools/build_notebook.py
python3 -m unittest discover -s tests -v
python3 tools/build_notebook.py --check

# 任意: torch/ultralyticsを導入した環境で、合成画像とランダム重みのCPU結合試験
python3 tools/smoke_training.py
# onnxを導入した環境では、同じ結合試験にONNX出力・checkerを追加
python3 tools/smoke_training.py --onnx-check
# 合成画像でepoch 1保存後に中断→ローカルrun削除→復元してepoch 2まで再開
python3 tools/smoke_training.py --drive-resume
```

ノートブックには関数定義と合成テストを通常のコードセルとして収めています。開発用src・合成テストを変更した後は再生成してください。ノートブックだけで動くことも合成テストで確認します。保存済みノートブックは実行回数null・全outputs空です。開発時にコードとノートブック全セルの構文も確認します。

実施済みの検証:

- 合成データによる87件の自動テストが成功。毎epoch保存・ランタイム消失後の復元・破損検出・設定不一致拒否・Drive保存失敗時の停止と前回の保存保持も確認しました。Driveを模したフォルダからのコピー、再利用、SHA-256不一致、CRC破損、不足ファイル、容量不足、symlink拒否とノートブックのDrive→Colab処理順も確認しました。進捗の完了数・表示頻度・サイズ不明時の表示・Colabの表示更新と、進捗付きCRC検証による破損検出も確認しました。CrowdHuman形式のカンマ入り画像IDの解析・3モード変換と、危険なIDの拒否も確認しました。ローカル取得先の分離、例外時の保存先復帰、symlink拒否も確認しました。HTTP 403・DNS失敗・既存ファイル破損・転送不足・中断・ログ書込失敗の診断、追記、秘密情報の除去も確認しました。Hugging Faceの固定URL、入力値制限、正常取得、既存ファイル再利用、HTML拒否、不完全レスポンスの削除も確認しました。このうちノートブック単体検査では、ノートブック内の12件の合成テストも実行して成功しています。変換・ignore・クリッピング・不正bbox・class・split・YAML・Git除外・不足/破損ファイル・ZIPパス検査・取得済みファイル再利用・再開ガード・比較CSV・ダウンロード許可対象・ONNX契約判定を確認しました。ONNX契約判定の単体テストはRuntimeをモックしています。
- Ultralytics 8.4.52 / PyTorch 2.12.0+cu130のCPUで、独自作成した合成画像とランダム初期化YOLO26による3モード各1 epochの学習、評価、PT/CSV出力、比較CSV生成が成功しました。学習出力に画像ファイルが生成されないことも確認しました。これは公式事前学習重みからのFine-tuningや実精度の検証ではありません。
- 合成画像と実際のYOLO26のCPU試験で、epoch 1の保存直後に中断し、ローカルrunを削除後、Driveを模したフォルダから復元してoptimizer付きでepoch 2まで再開できました。
- ランダム重みのYOLO26をopset 13・dynamic・FP32のONNXへ出力し、onnx.checkerが成功しました。
- Pythonとノートブック全コードセルの構文、埋め込みソースの一致、出力が空であること、Git差分の空白エラーと管理対象を確認しました。

未検証:

- Colabの実GPU、公式事前学習重みの取得、CrowdHuman実ファイルのダウンロード・展開・学習。実画像・実ODGTは開発環境へ取得していません。
- Google Driveの実マウント、実データのDrive保管・Colabへのコピー、files.downloadの実操作、実GPUのcheckpointからの再開。Drive保存・復元は合成データと一時フォルダで検証しています。
- ONNX Runtime実推論（開発環境に未導入）、ONNX Runtime Web、既存Webアプリ・追跡器との入出力互換性。ColabにはRuntimeをインストールし、エクスポート時の合成推論検査を実装しています。

GitHubへ反映する前に差分を確認し、以下でコミットします。データ・モデルは管理対象に含めないでください。

```bash
git add .gitignore .gitattributes .github README.md requirements.txt crowdhuman_finetuning.ipynb src tests tools
git commit -m "feat: add standalone CrowdHuman Colab notebook"
```

ローカル実ダウンロードの試行では、実行環境の名前解決が`[Errno -3] Temporary failure in name resolution`で失敗しました。HTTP応答には到達せず、データ取得は未完了です。診断ログはローカルの`crowdhuman/logs/download.jsonl`に保存されます。
