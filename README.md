# CrowdHumanでYOLO人物検出をファインチューニング

CrowdHumanのvbox（人物の見えている範囲）とhbox（頭部）を使い、YOLOをファインチューニングするColabノートブックです。1つのモデルで、`0: person`（vbox）と `1: head`（hbox）の2クラスを学習します。

[![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pj-hoakari/yolo_person_finetuning_colab/blob/main/yolo_person_finetuning.ipynb)

現在はGPU確認・Google Drive接続・2クラスのデータセット準備まで実装しています。セル1〜4を順番に実行してください。学習はまだ実行しません。

データ準備では、Driveの `crowdhuman/raw/archives` の4個のZIPと `raw/annotations` のODGT注釈を使用し、`/content/crowdhuman_vbox_yolo` に画像・YOLOラベル・`data.yaml`・集計JSONを作成します。元のtrain・val分割を維持します。既存のprepared tarは使用しません。

`tag=person` と無視指定を確認し、vboxをperson、hboxをheadとして変換します。枠を画像内に収め、無効な枠を除外します。無視枠をラベルから除く処理は、学習時の無視領域対応やCrowdHuman公式評価とは異なります。詳細はノートブックのセル4を参照してください。

同じ設定でデータ準備を再実行すると、展開済み画像を再利用できます。Colabローカルのデータが失われた場合は再準備が必要です。

学習処理は、小規模学習を挟まず設定エポック数で進め、各エポック終了時に再開用チェックポイントと学習結果をDriveへ保存する方針です。中断後は最後に正常保存できたエポックから再開します。学習・保存・再開処理は未実装です。
