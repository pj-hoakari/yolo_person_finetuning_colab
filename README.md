# CrowdHumanでYOLO人物検出をファインチューニング

CrowdHumanのvbox（人物の見えている範囲）とhbox（頭部）を使い、YOLOをファインチューニングするColabノートブックです。1つのモデルで、`0: person`（vbox）と `1: head`（hbox）の2クラスを学習します。

[![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pj-hoakari/yolo_person_finetuning_colab/blob/main/yolo_person_finetuning.ipynb)

GPU確認・Google Drive接続・2クラスのデータセット準備・事前学習モデル取得・学習まで実装しています。重み保存・自動再開は現在無効です。セル1〜6を順番に実行してください。セル6の実行前にエポック数・画像サイズ・バッチサイズ・RUN_NAMEを設定してください。

データ準備では、Driveの `crowdhuman/raw/archives` の4個のZIPと `raw/annotations` のODGT注釈を使用し、`/content/crowdhuman_vbox_yolo` に画像・YOLOラベル・`data.yaml`・集計JSONを作成します。元のtrain・val分割を維持します。既存のprepared tarは使用しません。

`tag=person` と無視指定を確認し、vboxをperson、hboxをheadとして変換します。枠を画像内に収め、無効な枠を除外します。無視枠をラベルから除く処理は、学習時の無視領域対応やCrowdHuman公式評価とは異なります。詳細はノートブックのセル4を参照してください。

同じ設定でデータ準備を再実行すると、展開済み画像を再利用できます。Colabローカルのデータが失われた場合は再準備が必要です。

セル6は小規模学習を挟まず、設定したエポック数で毎回、事前学習重みから学習します。エポックごとのDrive保存・FP32チェックポイント保存・`latest.json` 更新・自動再開を削除し、Ultralytics標準の重み保存も最終エポックを含めて無効にしています。NaN/Inf検出時の停止、学習率・AMP・ウォームアップなどの学習設定は維持します。CSV・学習設定・図は `/content/crowdhuman_training` にローカル出力されます。出力先が既存の場合は新しい番号付きフォルダを使います。学習済みモデルは `TRAINING.model` に残りますが、ランタイムを失うと復元できません。Driveの過去の保存済みデータは削除しません。

セル5はUltralytics `8.4.0` を導入し、`yolo26n.pt` をDriveの `crowdhuman_vbox_ft/weights` に保存します。保存済みの重みは再利用し、重みのハッシュと環境情報も記録します。

学習ログのアーカイブ・ダウンロードはセル7を実行してください。日時付きのCSV・学習設定（YAML/TXT）・最終/ベスト指標のJSONサマリー・図をZIPにまとめてダウンロードします。現在の `TRAINING.save_dir` を使用します。重みは含めず、DriveへのZIP保存も行いません。ランタイムをリセットするとローカルのログは失われます。以前Driveに保存した実験のログを取得する場合は `DOWNLOAD_RUN_DIR` に保存先を指定してください。
