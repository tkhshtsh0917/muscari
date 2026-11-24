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

## 検索学習

`train_retriever.py`は、Sentence Transformersで疎検索モデルを学習し、学習の前後で並べ替え用データセットの候補を順位付けします。`retrieval`の追加依存が必要です。2種類のヘッドを同じデータ、損失、スケジュールで学習できるので、両者を比べられます。

- `--head sae`は、書き出したSAEの潜在を語彙として使います。`muscari.retrieval`を通します。
- `--head mlm`は、SPLADEと同じく、言語モデルのヘッドを通してトークナイザの語彙を使います。エンコーダにそのヘッドが残っている必要があります。

```bash
python examples/train_retriever.py \
  --head sae --sae runs/ruri-v3-30m-sae-trial \
  --query-prompt "検索クエリ: " --document-prompt "検索文書: " \
  --dataset cl-nagoya/ruri-v3-dataset-ft --subsets jsquad \
  --query-column anc --positive-column pos --negative-column neg \
  --benchmark jqara --max-steps 1000 \
  --out runs/ruri-v3-30m-retriever-trial
```

損失は、バッチ内のネガティブとクエリごとに1件のハードネガティブを使う対照損失に、クエリと文書へのFLOPS正則化を加えたものです。正則化の重みは、最初の30%のステップで0から二次関数的に上げます。学習率、正則化の重み、このスケジュールは、[sae-splade-ja-310m](https://huggingface.co/mahiyama/sae-splade-ja-310m)のモデルカードが学習の第1段階について書いている値です。バッチサイズ、長さの上限、ステップ数は小さめの既定値で、カードの値ではありません。

スクリプトは、学習の前後のnDCG@10、MRR@10、クエリ・文書1件あたりの非ゼロ次元数、FLOPSを`results.json`に書き出し、モデルを保存します。評価では、クエリのベクトルをデバイス上に置いたまま、文書を少しずつ符号化してその場でスコアを計算し、スコアと次元ごとの件数だけを残します。学習前にはほぼ密になりうる語彙の大きさの出力を、コレクション全体の分だけ持つことはありません。アクセラレータ上で疎テンソルを組み立てることもありません。学習中は、全ステップの10分の1ごとにチェックポイントを保存します。同じ`--out`で起動し直すと、学習前の評価を使い回し、最後のチェックポイントから続けます。SAEのヘッドの学習前の値は、同じクエリに対する`muscari probe-sae`の`sparse.plain`と一致します。SAEヘッドの学習はfloat32で行ってください。`--bf16`を付けると、float32の2本より明らかに低い結果になりました。`sbintuitions/modernbert-ja-30m`では、語彙型ヘッドが学習前からほぼ密で、この既定の設定では数百ステップ以内に崩壊します。どちらの結果も[設計ガイド](../docs/ja/design.md#小規模な検索学習)にあります。
