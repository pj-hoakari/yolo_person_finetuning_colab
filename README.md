# CrowdHumanを用いたYOLO追加学習

Google Colabの一時ストレージだけでCrowdHumanを取得・変換し、YOLO26nの追加学習と評価を行います。`crowdhuman_finetuning.ipynb` をColabで開き、設定を確認して上から実行します。全処理と合成テストは、ノートブック内の通常のPythonセルとして収めています。外部のsrcファイルやテストファイルの取得・生成は不要です。

[Colabで開く（GitHubへ反映後）](https://colab.research.google.com/github/aira0110/yolo_person_finetuning_colab/blob/main/crowdhuman_finetuning.ipynb)

## GitHubでの管理

成果物は、このリポジトリのルートに置いています。`crowdhuman_finetuning.ipynb`だけをColabへアップロードして実行できます。GitHubで実装を変更する場合は、読みやすい差分を確認できる`src/`と`tools/notebook_tests.py`を編集し、`python3 tools/build_notebook.py`でノートブックのコードセルへ反映してください。実行時のsrc importやPythonファイル展開はありません。

`.github/workflows/tests.yml`は、push/PR時に合成データだけで変換・ノートブック単体動作・生成内容の一致を検証します。CrowdHuman取得、モデル学習、Artifacts・Releasesへの保存は行いません。`requirements.txt`はColab用で、CIではPillowとPyYAMLのみを導入します。

## 最初に確認すること

- [CrowdHuman公式配布ページ・利用条件](https://www.crowdhuman.org/download.html)では、非商用の研究・教育目的に用途を限定し、画像の再配布を禁止しています。本実装ではアノテーションも非公開として扱います。
- ColabはGoogleのクラウド環境です。今回調査した配布ページだけから、あらゆる利用条件でクラウド処理が許されるとは判断していません。適用される条件を確認し、クラウド処理が禁止される場合はColabを使わずローカル環境用に別途移植してください。実データのローカル実行をこのコードの保存先制限で迂回しないでください。
- [Ultralyticsのライセンス](https://www.ultralytics.com/license)はAGPL-3.0とEnterpriseの選択肢を案内しています。モデル・ライブラリの条件を確認してください。商用・非公開のWebアプリへの移行可否は、CrowdHumanの用途制限と独立に確認が必要です。
- 学習済みモデルにもデータ由来の情報が含まれる可能性があります。重みのダウンロードは公開の許可を意味しません。

## 参考実装との差分

[参考ノートブック](https://github.com/KasumiMercury/yolo_person_finetuning_colab/blob/main/yolo_person_finetuning.ipynb)を調査しました。

| 項目 | 参考実装 | 今回の実装 |
|---|---|---|
| モデル | yolo26n.pt（sへ変更可） | 同じモデルを優先 |
| データ | personに絞ったCOCO YOLO ZIP | 公式CrowdHuman ODGTを3モードへ変換 |
| 保存先 | DriveとColab SSD | `/content/crowdhuman/`のみ |
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
src/dataset.py              # 公式取得、ZIP/ODGT/画像検証、安全な展開
src/converter.py            # vbox / hbox / both変換、ラベル検査
src/trainer.py             # 学習設定、再開、実験情報
src/evaluator.py            # 評価、比較CSV、ONNX検査、手動ダウンロード
src/privacy.py              # 保存先制限、外部連携停止、ログ制御
tests/test_converter.py     # 実データ不要の合成テスト
tools/build_notebook.py     # src/とノートブックの通常コードセルを同期
tools/notebook_tests.py     # ノートブックへ直接収める合成テスト
.github/workflows/tests.yml # 実データを使わないGitHub Actions
tools/smoke_training.py     # 合成画像とランダム重みによる任意CPU結合テスト
requirements.txt
.gitignore
```

## Colabの使い方

1. ノートブックをColabへアップロードし、GPUランタイムを選びます。「編集 → ノートブックの設定 → 保存時にコードセルの出力を除外する」も有効にします。共有前に全出力を削除してください。
2. セル2〜5で環境・依存・学習設定を準備します。利用条件・Colabでの処理可否を確認後、設定セルの`TERMS_REVIEWED=True`にします。認証情報はコードへ入力しません。
3. 最初は`DEBUG_MODE=True`、`EPOCHS=1`、必要に応じてbatchを下げて確認します。DEBUGは各splitの先頭`DEBUG_LIMIT`画像を使います。全モードで同じ画像を選びます。ダウンロード容量・入力全体の検証量は減りません。
4. セル6〜12で取得・展開・画像検証・変換・合成テストを実行します。画像IDやアノテーション自体は出力しません。
5. セル13〜17でモデル準備、学習、評価、比較、モデル出力を行います。GPUがない場合は警告して停止します。意図的なCPUテストだけ`ALLOW_CPU=True`を指定できます。
6. セル18で`DOWNLOAD_NAMES`に必要なファイルを列挙し、`DO_DOWNLOAD=True`としてそのセルを明示実行します。
7. 必要な重みの取得を確認後、セル19の削除フラグを必要なものだけ有効にします。

**ランタイム終了・リセットで、画像、ラベル、ログ、チェックポイント、exportsも失われます。** Driveへの自動保存はありません。ノートブックの保存は学習済み重みの保存にはなりません。

### 公式取得と手動フォールバック

必要なファイルは次の6つです。

- `CrowdHuman_train01.zip` / `CrowdHuman_train02.zip` / `CrowdHuman_train03.zip`
- `CrowdHuman_val.zip`
- `annotation_train.odgt` / `annotation_val.odgt`

実行時に公式ページが案内するGoogle Driveの公開リンクを読み取り、通常の公開HTTPダウンロードを試みます。これはDriveのマウントではありません。認証・アクセス制限・確認画面への回避処理はありません。今回の開発環境から配布ファイルを実際に取得できることは未検証です。

自動取得に失敗した場合だけ、公式から正規に入手したファイルをColabのアップロードUIで配置します。自動取得不能を既に確認した場合は`AUTO_DOWNLOAD=False`で手動経路に入れます。大容量ZIPのブラウザアップロードはRAMを使うため、必要なら1つずつアップロードしてセルを再実行してください。アップロード対象を間違えないでください。

アップロード中のファイルは`raw/`に置かれ、検証後にarchives/annotationsへ移動します。正常な既存ファイルは再ダウンロードしません。既存ファイルが破損している場合は停止し、無言で置換しません。Colabのファイル欄で破損した対象を明示的に削除・交換して再実行します。

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

bboxを画像境界へクリッピングしてからYOLOの中心x/中心y/幅/高さへ正規化します。ゼロ・負幅、非数、非有限値、クリッピング後の空矩形は除外します。モード別の`conversion.json`にsplit別の採用・ignore・不正・クリッピング件数を記録します。不正ODGT行や危険なID、画像不足、split重複、壊れたラベル/YAMLは停止条件です。

**標準YOLOラベルはignore領域を表現できません。** 除外された人物・頭部・maskの領域は学習時に背景と扱われ、評価時にもignore領域の検出が誤検出になり得ます。空ラベル画像も背景画像として残します。これはCrowdHuman公式のignore対応評価や公式指標の再現ではなく、変換後のYOLOラベルに対する評価です。画像を加工してignore領域を隠す処理は行っていません。

### 比較と再開

`MODES_TO_RUN=['vbox', 'hbox']`で条件を揃えて2モデルを順番に学習します。bothを追加する場合は`['vbox', 'hbox', 'both']`を使います。各モデルを同じ初期モデルから独立に初期化します。

結果は`exports/comparison.csv`です。Precision、Recall、mAP50、mAP50-95、学習セッション累計秒、推論ms/画像、重みMiBを保存します。bothはクラス別の行も追加します。`matched_conditions=False`なら学習設定・データ指紋・依存・評価条件・GPU名が揃っていません。vboxとhboxは検出対象が異なり、mAPの大小だけで優劣を決められません。1モードだけでもCSVを出せますが、比較には複数の実験が必要です。

推論速度はbatch=1・FP32のvalidation内部のinference時間です。前処理・後処理・ブラウザ・追跡は含みません。学習時間はモデル準備を含み、中断・再開分も合算します。early stoppingにより実際のepoch数が異なることがあります。各`results.csv`も確認してください。

同じセッションの中断から再開するには、同一設定・RUN_NAMEで`RESUME=True`として学習セルを再実行します。last.ptにoptimizerと有効なepochがあること、保存した設定・変換指紋・依存バージョンが一致することを要求します。完了時にoptimizerを削除したcheckpointは再開対象になりません。既存実験への新規学習の上書きは拒否します。

DEBUGから本番への移行やデータ変更時は、該当`processed/<mode>`をColabファイル欄から明示削除して再変換し、別のRUN_NAMEで新規学習します。rawは再利用できます。モードごとに実験を残し、比較CSVは保存済み条件が揃わない行も一致フラグをFalseにして表示します。

### 出力とWebアプリへの接続

`exports/<mode>_<RUN_NAME>.pt`、`.onnx`、`_training.csv`、`comparison.csv`がダウンロード対象です。DEBUGでは実験名に`_debug`が付きます。runディレクトリ一式をZIPにしないため、画像・ラベルの混入を避けられます。`files.download()`を呼ぶのはセル18でユーザーが明示指定した場合だけです。

ONNXはopset 13、dynamic、FP32、simplify=Falseで出力します。`onnx.checker`とONNX Runtime CPUの合成ゼロ画像で検査し、`*_onnx_interface.json`に入出力名・dtype・shape・クラスを記録します。既存アプリのコード・モデル仕様はこの作業場所に提供されていないため、ブラウザ互換性は未検証です。

`WEBAPP_CONTRACT`にはアプリが実際に要求する辞書を指定できます。生成レポートの`interface`と同じ構造（`input: {name, dtype, shape}`、`outputs: [{name, dtype, shape}]`）で完全一致を検査します。未指定なら`application_contract: unverified`です。一致してもブラウザでの結合試験の代替にはなりません。

アプリ側ではRGB、float32、NCHW、0〜1への正規化、letterbox、元画像への座標復元、confidence・class ID、ONNX Runtime Webのopset対応を確認してください。YOLO26のend-to-end出力は、従来のYOLOの生出力を前提とするデコーダと異なる可能性があります。実際の出力shapeと座標形式を確認し、NMSの有無と追跡器への入力も合わせて検証してください。

## プライバシーとGit

実データの保存先はコードで`/content/crowdhuman/`配下に制限します。範囲外へのsymlinkも拒否します。Driveマウント、Hubアップロード、外部画像API、GitHub Releases/Artifacts/Actionsへの保存処理はありません。同期・外部実験追跡を設定で無効にし、HUB/Platform/analytics等のintegration callback登録も停止します。学習・検証のplot、画像表示、予測画像保存を使いません。大量ログはランタイム内のconsole.log等に保存し、ノートブックには要約だけを表示します。

これはOSのネットワーク遮断ではありません。パッケージ・公式重み・公式データの取得にはネットワークを使います。ユーザーが別途外部連携を有効にしないでください。ノートブックを共有する前に、出力除去とステージ済みファイルを確認してください。

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
```

ノートブックには関数定義と合成テストを通常のコードセルとして収めています。開発用src・合成テストを変更した後は再生成してください。ノートブックだけで動くことも合成テストで確認します。保存済みノートブックは実行回数null・全outputs空です。開発時にコードとノートブック全セルの構文も確認します。

実施済みの検証:

- 合成データによる43件の自動テストが成功。このうちノートブック単体検査では、ノートブック内の12件の合成テストも実行して成功しています。変換・ignore・クリッピング・不正bbox・class・split・YAML・Git除外・不足/破損ファイル・ZIPパス検査・取得済みファイル再利用・再開ガード・比較CSV・ダウンロード許可対象・ONNX契約判定を確認しました。ONNX契約判定の単体テストはRuntimeをモックしています。
- Ultralytics 8.4.52 / PyTorch 2.12.0+cu130のCPUで、独自作成した合成画像とランダム初期化YOLO26による3モード各1 epochの学習、評価、PT/CSV出力、比較CSV生成が成功しました。学習出力に画像ファイルが生成されないことも確認しました。これは公式事前学習重みからのFine-tuningや実精度の検証ではありません。
- ランダム重みのYOLO26をopset 13・dynamic・FP32のONNXへ出力し、onnx.checkerが成功しました。
- Pythonとノートブック全コードセルの構文、埋め込みソースの一致、出力が空であること、Git差分の空白エラーと管理対象を確認しました。

未検証:

- Colabの実GPU、公式事前学習重みの取得、CrowdHuman実ファイルのダウンロード・展開・学習。実画像・実ODGTは開発環境へ取得していません。
- Colabの大容量アップロードUIとfiles.downloadの実操作、実GPUのcheckpointからの再開。再開分岐と設定変更拒否はモックによるテストです。
- ONNX Runtime実推論（開発環境に未導入）、ONNX Runtime Web、既存Webアプリ・追跡器との入出力互換性。ColabにはRuntimeをインストールし、エクスポート時の合成推論検査を実装しています。

GitHubへ反映する前に差分を確認し、以下でコミットします。データ・モデルは管理対象に含めないでください。

```bash
git add .gitignore .gitattributes .github README.md requirements.txt crowdhuman_finetuning.ipynb src tests tools
git commit -m "feat: add standalone CrowdHuman Colab notebook"
```
