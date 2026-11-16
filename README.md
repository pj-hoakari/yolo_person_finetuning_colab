# CrowdHumanでYOLO人物検出をファインチューニング

CrowdHumanのvbox（人物の見えている範囲）とhbox（頭部）を使い、YOLOをファインチューニングするColabノートブックです。1つのモデルで、`0: person`（vbox）と `1: head`（hbox）の2クラスを学習します。

[![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pj-hoakari/yolo_person_finetuning_colab/blob/main/yolo_person_finetuning.ipynb)

GPU確認・Google Drive接続・2クラスのデータセット準備・事前学習モデル取得・学習・Driveへの自動保存・再開まで実装しています。セル1〜6を順番に実行してください。セル6の実行前にエポック数・画像サイズ・バッチサイズ・RUN_NAMEを設定してください。

データ準備では、Driveの `crowdhuman/raw/archives` の4個のZIPと `raw/annotations` のODGT注釈を使用し、`/content/crowdhuman_vbox_yolo_ignore_masked` に画像・YOLOラベル・`data.yaml`・集計JSONを作成します。元のtrain・val分割を維持します。既存のprepared tarは使用しません。注釈2ファイルと画像ZIP4個がDriveに無い場合は、セル3がHugging Faceのミラー（ミラー間で一致したサイズ・SHA-256で検証）から取得します。画像ZIPは合計約10.9GiBで、中断しても再実行で続きから取得します。注釈は公式のGoogle Driveも予備として使用します。Driveの空き容量が足りない場合は取得前に停止します。

`tag=person` と無視指定を確認し、vboxをperson、hboxをheadとして変換します。枠を画像内に収め、無効な枠を除外します。ignore人物・tag=maskはfbox（無効ならvbox・hbox）、ignore頭部はhboxをYOLOのLetterBoxと同じRGB `(114, 114, 114)` のグレーで塗りつぶし、ラベルからは従来どおり除外します。塗りつぶしは損失を無効化する処理やCrowdHuman公式評価とは異なります。詳細はノートブックのセル4を参照してください。

塗りつぶし対象の原画像をローカルの `.original_images` に保持し、再実行時も原画像から加工します。Driveの元画像は変更しません。新しい前処理を適用するにはセル3〜4を再実行し、学習時は新しいRUN_NAMEを使用してください。同じ設定では展開済み原画像を再利用できます。Colabローカルのデータが失われた場合は再準備が必要です。

セル6は各エポック終了時に、学習モデル状態・EMA・オプティマイザーとAMP scaler（元のデータ型と値を維持）をDriveの `crowdhuman_vbox_ft/runs/<RUN_NAME>/epochs` に自動保存します。全ファイルのコピーとSHA-256確認後にだけ `latest.json` を更新し、再実行時はそのエポックの次から再開します。NaN/Inf検出や保存失敗時は停止して再開先を維持します。学習率・AMP・ウォームアップなどの設定は変更していません。既存の設定やデータと異なる場合は新しいRUN_NAMEを指定してください。正常終了時の成果物は `final` に保存します。旧形式のFP16チェックポイントで失われた精度は復元できません。

セル5はUltralytics `8.4.0` を導入し、`yolo26n.pt` をDriveの `crowdhuman_vbox_ft/weights` に保存します。保存済みの重みは再利用し、重みのハッシュと環境情報も記録します。

学習ログのアーカイブ・ダウンロードはセル7を実行してください。日時付きのCSV・学習設定（YAML/TXT）・最終/ベスト指標のJSONサマリー・図をZIPにまとめてダウンロードします。現在の `TRAINING.save_dir` を使用し、ランタイム再接続後はDriveの実験フォルダを参照します。重みは含めず、DriveへのZIP保存も行いません。ランタイムをリセットするとローカルのログは失われますが、Driveの正常保存分は取得できます。以前Driveに保存した実験のログを取得する場合は `DOWNLOAD_RUN_DIR` に保存先を指定してください。

評価時のEMAモデルのFP16変換によるBatchNorm統計のオーバーフローを避けるため、評価にはFP32を使用します。保存時は再開用・最終推論用ともに型変換せず、元のデータ型と値を維持します。学習中のAMP・学習率・ウォームアップ設定は維持します。旧版で既にInfになったEMAは復元できないため、更新したセル6でTrainerを作り直し、正常な保存分または1エポック目から開始してください。

旧版の黒塗りを変更する場合はセル4を再実行してください。色だけの変更は保持した原画像から再生成でき、既存の展開済み原画像を再利用します。学習には新しいRUN_NAMEを使用してください。
