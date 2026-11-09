# CrowdHumanでYOLO人物検出をファインチューニング

CrowdHumanのvbox（人物の見えている範囲）とhbox（頭部）を使い、YOLOをファインチューニングするColabノートブックです。1つのモデルで、`0: person`（vbox）と `1: head`（hbox）の2クラスを学習します。

[![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pj-hoakari/yolo_person_finetuning_colab/blob/main/yolo_person_finetuning.ipynb)

GPU確認・Google Drive接続・2クラスのデータセット準備・事前学習モデル取得・学習と保存・再開まで実装しています。セル1〜6を順番に実行してください。セル6の実行前にエポック数・画像サイズ・バッチサイズ・RUN_NAMEを設定してください。

データ準備では、Driveの `crowdhuman/raw/archives` の4個のZIPと `raw/annotations` のODGT注釈を使用し、`/content/crowdhuman_vbox_yolo` に画像・YOLOラベル・`data.yaml`・集計JSONを作成します。元のtrain・val分割を維持します。既存のprepared tarは使用しません。

`tag=person` と無視指定を確認し、vboxをperson、hboxをheadとして変換します。枠を画像内に収め、無効な枠を除外します。無視枠をラベルから除く処理は、学習時の無視領域対応やCrowdHuman公式評価とは異なります。詳細はノートブックのセル4を参照してください。

同じ設定でデータ準備を再実行すると、展開済み画像を再利用できます。Colabローカルのデータが失われた場合は再準備が必要です。

セル6は小規模学習を挟まず、設定した総エポック数で学習します。各エポックの再開用チェックポイント・その時点のbest重み・累積CSVなどをDriveの `crowdhuman_vbox_ft/runs/<RUN_NAME>/epochs` に保持し、コピー確認後に `latest.json` を更新します。同じ設定での再実行は最後の正常保存分の次のエポックから再開します。早期終了は無効です。設定やデータを変える場合は新しいRUN_NAMEを指定してください。正常終了時の成果物は `final` に保存します。

セル5はUltralytics `8.4.0` を導入し、`yolo26n.pt` をDriveの `crowdhuman_vbox_ft/weights` に保存します。保存済みの重みは再利用し、重みのハッシュと環境情報も記録します。
