# 例

[English](README.md)

## オフラインでの動作確認

`train_tiny.py`は、ランダムな重みの2層BERTと、数百行の合成したクエリ・文書データを一時ディレクトリに作ります。そこで`muscari train-sae`を実行し、書き出したSAEを`muscari eval-sae`で評価します。ダウンロードは行わず、CPUでも数秒で終わります。

```bash
python examples/train_tiny.py --trainer topk
python examples/train_tiny.py --trainer saelens    # "saelens"の追加依存が必要です
python examples/train_tiny.py --trainer topk --workdir runs/tiny    # ファイルを残す場合
```

開発用のチェックアウトでは、`mise run smoke`で両方のトレーナーを実行できます。

スクリプトは、学習ログ、学習後のホールドアウト評価、書き出したファイルから測り直した指標を出力します。コマンドライン引数の動く組み合わせと、1回の実行で作られるファイルを確認するために使ってください。エンコーダはランダムなので、数値はSAEの品質を表しません。

## 実際のエンコーダ

ruri-v3での実行方法は[README](../README.ja.md#クイックスタート)にあります。長時間の学習を始める前に、小さなモデルと短いトークン数で、手元の環境でパイプラインが動くことを確認できます。

```bash
muscari train-sae \
  --trainer topk \
  --model cl-nagoya/ruri-v3-30m \
  --dataset cl-nagoya/ruri-v3-dataset-ft \
  --subsets jsquad \
  --columns "anc=検索クエリ: " "pos=検索文書: " "neg=検索文書: " \
  --num-latents 8192 \
  --total-tokens 5000000 \
  --dead-after-tokens 1000000 \
  --out runs/ruri-v3-30m-sae-trial
```

このリポジトリのmise環境では、検査中に何もダウンロードしないよう、Hugging Face Hubをオフラインにしています。その中で実行するときは、コマンドの前に`mise exec -- env HF_HUB_OFFLINE=0 uv run --frozen --no-sync`を付けてください。`mise exec`の前に変数を書いても効きません。

MPSを使うApple M5 Proでは、このコマンドは200秒で終わりました。モデルとデータセットのダウンロード、ホールドアウト評価を含む時間です。得られた数値は[設計ガイド](../docs/ja/design.md#実際のエンコーダでの試走)にあります。
