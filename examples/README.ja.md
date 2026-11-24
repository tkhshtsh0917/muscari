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

`train_retriever.py`は、Sentence Transformersで疎検索モデルを学習し、学習の前後で評価します。`retrieval`の追加依存が必要です。3種類のヘッドを同じデータ、損失、スケジュールで学習できるので、互いに比べられます。

- `--head sae`は、書き出したSAEの潜在を語彙として使います。`muscari.retrieval`を通します。`--trainable-sae`を付けない限り、SAEのエンコーダは固定です。
- `--head vocab`は、言語モデルのヘッドを通してトークナイザの語彙を使います（`vocabulary_head.py`）。SAEヘッドとまったく同じトークン、つまり枠のトークンとプロンプトのトークンを除いたものを集約するので、2つのヘッドの違いは辞書だけです。`--shift`はReLUの前にロジットから定数を引き、`--fan-out`はトークンごとにその数のロジットだけを残します。どちらも、SPARSEUP（[arXiv 2611.09680](https://arxiv.org/abs/2611.09680)）が学習を疎な状態から始めるために入れた変更です。`--shift auto`は、学習文書`--shift-texts`件の内容トークンのロジットの`--shift-quantile`分位点（既定0.999）を推定してずらし幅にします。ずらし幅とファンアウトがなく、すべてのトークンを集約すると、このヘッドは`SpladePooling`と一致します。ファンアウトの勾配が残したロジットにだけ流れるのは、このプロジェクトの選択です。
- `--head mlm`は、Sentence Transformersの`SpladePooling`をそのまま使います。プロンプトと特殊トークンも集約に入ります。

`vocab`と`mlm`では、エンコーダに言語モデルのヘッドが残っている必要があります。

```bash
python examples/train_retriever.py \
  --head sae --sae runs/ruri-v3-30m-sae-trial \
  --query-prompt "検索クエリ: " --document-prompt "検索文書: " \
  --dataset cl-nagoya/ruri-v3-dataset-ft --subsets jsquad \
  --query-column anc --positive-column pos --negative-column neg \
  --eval-benchmark jqara --eval-split dev --max-steps 1000 \
  --out runs/ruri-v3-30m-retriever-trial
```

損失は、バッチ内のネガティブとクエリごとに1件のハードネガティブを使う対照損失に、クエリと文書へのFLOPS正則化を加えたものです。正則化の重みは、最初の30%のステップで0から二次関数的に上げます。学習率、正則化の重み、このスケジュールは、[sae-splade-ja-310m](https://huggingface.co/mahiyama/sae-splade-ja-310m)のモデルカードが学習の第1段階について書いている値です。バッチサイズ、長さの上限、ステップ数は小さめの既定値で、カードの値ではありません。`--mini-batch-size`を付けると、各バッチをGradCache（`CachedSpladeLoss`）で分割して符号化します。勾配は同じで、メモリは少なくて済みます。

学習中は、全ステップの10分の1ごとにチェックポイントを保存します。同じ`--out`で起動し直すと、学習前の評価を使い回し、最後のチェックポイントから続けます。`results.json`には、ヘッドの設定、学習ログ、学習前後の評価が入り、モデルは`model/`に保存されます。SAEのヘッドの学習前の値は、同じクエリに対する`muscari probe-sae`の`sparse.plain`と一致します。SAEヘッドの学習はfloat32で行ってください。`--bf16`を付けると、float32の2本より明らかに低い結果になりました。`sbintuitions/modernbert-ja-30m`では、ずらし幅のない語彙型ヘッドが学習前からほぼ密で、この既定の設定では数百ステップ以内に崩壊します。どちらの結果も[設計ガイド](../docs/ja/design.md#小規模な検索学習)にあります。

## 評価

`sparse_eval.py`は、Sentence Transformersの`SparseEncoder`ならどれでも評価します。`train_retriever.py`が保存したモデル、学習前のSAEヘッド（`--sae`）、公開済みの疎検索モデルのいずれでも構いません。`train_retriever.py`は、オプション名の頭に`--eval-`を付けて同じ評価を行います。

```bash
python examples/sparse_eval.py --model runs/ruri-v3-30m-retriever-trial/model \
  --benchmark jqara --split test \
  --retrieval mteb/MIRACLJaRetrievalLite mteb/JaCWIRRetrievalLite \
  --caps full 64/400 32/256 --out runs/ruri-v3-30m-retriever-trial/eval.json
```

- 並べ替えでは、クエリごとにそのクエリの候補を順位付けします。JQaRAでは、ベンチマーク自身の評価スクリプトと同じく、文書を`"{title} {text}"`にします（`--without-title`でタイトルを外します）。`--queries`で標本を取らない限り、分割のクエリをすべて使います。設定を選ぶには`dev`、最終的な数値には`test`を使います。
- 検索では、MTEB形式（構成`corpus`、`queries`、`qrels`、またはその3つの`.jsonl`を置いたディレクトリ）のコーパス全体を順位付けし、nDCG@10、MRR@10、recall@100を報告します。コーパスを少しずつエンコーダに通し、クエリごとの上位100件に順次まとめるので、コーパスのベクトルもスコア行列全体も保持しません。
- `--caps QUERY/DOCUMENT`は、推論時に各ベクトルの大きい重みだけを残します。`full`はすべて残します。どの指標にも、クエリと文書1件あたりの非ゼロ次元数とFLOPS（`sum_j p_q(j) p_d(j)`）が付きます。

結果には、クエリごとの値もそのidをキーにして入ります。モジュールがSentence Transformersに含まれない公開モデルには`--trust-remote-code`が必要で、これはそのリポジトリのコードを実行します。先にそのコードを読んでください。これらのスクリプトが保存したローカルのディレクトリは、Pythonファイルを含まない限り、このオプションなしで読み込めます。

`compare_runs.py`は、2つの条件を同じクエリの上で比べます。条件ごとに複数の結果ファイル（例えば学習のシードごとに1つ）を渡すと、まずクエリごとに平均します。2つの平均、差、クエリについての対応のあるブートストラップによる95%区間、符号反転による並べ替え検定のp値を報告します。

```bash
python examples/compare_runs.py \
  --a runs/sae-seed*/results.json --b runs/vocab-seed*/results.json \
  --task reranking --cap full --metric ndcg@10
```

## SAEの取り込み

`import_sae.py`は、ほかで学習したTop-K型のSAEをmuscariの形式に包み、`eval-sae`、`inspect-sae`、`probe-sae`と検索用ヘッドで使えるようにします。元のファイルは、muscariと同じテンソル名と形（`encoder.weight`、`encoder.bias`、`decoder.weight`、`decoder.bias`）を持つsafetensorsファイル1つである必要があります。[mahiyama/modernbert-ja-310m-sae](https://huggingface.co/mahiyama/modernbert-ja-310m-sae)のモデルカードは、そのSAEがこの形であると書いています。muscariはデコーダのバイアスを`encoder.bias`に畳み込んで持ちます。畳み込んでいないエンコーダのバイアスを持つファイルには`--fold-decoder-bias`を付けます。`--column`で渡す列とプレフィックスは、評価用のコマンドがテキストを読む方法を決めるだけです。

```bash
python examples/import_sae.py --weights sae.safetensors --k 16 \
  --base-model sbintuitions/modernbert-ja-310m --max-length 512 \
  --column "anc:query=検索クエリ: " --column "pos:document=検索文書: " \
  --out runs/imported-sae
```
