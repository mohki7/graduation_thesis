# 回帰不連続デザインによる2023年のMLBのルール改正の効果検証

## 研究概要
MLB（Major League Baseball）の人気低下が近年話題となっている。その主な原因は「試合時間が長いこと」とされており、2023年はその対策の一つとしてルール改正が行われた。変更されたルールは複数存在するが、本論文ではそのうちの「ピッチクロック」が観客動員者数に与える影響を、統計的因果推論手法の一つである回帰不連続デザインによって検証する。

## プログラム内容
### データ収集
データはMLB Stats APIを用いて取得した。

### アルゴリズム
現時点ではモデルとして線形回帰を用いた回帰不連続デザインと、周期回帰を用いた回帰不連続デザインを実装している。今後、SARIMAや状態空間モデルの適用も検討中。

## お問い合わせ
何か質問や連絡したいことがあれば、X(Twitter)のDMからお願いします。
X(Twitter): https://twitter.com/A7_data
