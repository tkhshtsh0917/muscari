# muscari

[English](README.md)

テキストエンコーダ向けのスパースオートエンコーダ（SAE）です。エンコーダの隠れ状態でSAEを学習し、SAE型の学習型疎検索の語彙として使います。

muscariは、[ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m)のような凍結したテキストエンコーダにテキストを通し、本文トークンごとの隠れ状態でTop-K SAEを学習します。得られるのは数万本の方向からなる辞書です。検索モデルは、トークナイザの語彙の代わりにこの方向の上でクエリと文書を採点できます。SAE-SPLADEやSPLAREの考え方です。

現在は初期版（`0.1.0.dev0`）です。SAEの学習・書き出し・評価と、書き出したSAEを疎検索のエンコーダにするSentence Transformers向けのヘッドを扱います。このヘッドで検索モデルを学習したことは、まだありません。[今後の予定](#今後の予定)を参照してください。

## 専用のツールにした理由

- **エンコーダはデコーダ型LLMではありません。** SAELensの学習ランナーはTransformerLens経由でモデルを読み込みます。muscariはHugging Faceのエンコーダを`AutoModel`で読み込み、活性のテンソルをそのままトレーナーへ渡します。
- **学習するのは本文トークンだけです。** パディング、特殊トークン、指示プレフィックス（`検索クエリ: `、`query: `）のトークンは学習前に除きます。辞書は、疎検索のヘッドがプーリングする位置だけに使われます。
- **活性はその場で計算します。** エンコーダの推論は軽いので、ディスクには保存しません。768次元の活性を10億トークン分保存すると約3 TBになります。
- **トレーナーは二つ、出力形式は一つです。** 自前の学習ループとSAELensバックエンドが同じ活性ストリームを読み、同じファイルを書きます。結果は同じ評価コードで比べられます。
- **逆伝播は疎です。** 勾配を受け取る潜在はトークンあたり`k`個だけです。muscariは`(トークン数, 潜在数)`の密な勾配を作らず、選ばれた要素だけを通して逆伝播します。2スレッドのCPUで潜在65,536個、幅768、2,048トークンの場合、補助損失を含む1ステップは2.2秒でした。密な自動微分では11.0秒です。測定条件は[設計ガイド](docs/ja/design.md#5-疎な逆伝播)にあります。

## インストール

muscariはPyPIに登録していません。リポジトリからインストールします。

```bash
pip install "git+https://github.com/tkhshtsh0917/muscari.git"

# SAELensバックエンドも使う場合
pip install "muscari[saelens] @ git+https://github.com/tkhshtsh0917/muscari.git"

# Sentence Transformers向けの検索ヘッドも使う場合
pip install "muscari[retrieval] @ git+https://github.com/tkhshtsh0917/muscari.git"
```

Python 3.11以降、PyTorch 2.6以降、Transformers 4.48以降が必要です。検索ヘッドにはSentence Transformers 6.3以降が必要で、こちらはTransformers 5を要求します。開発環境は[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)を参照してください。

## クイックスタート

ruri-v3の最終層で、ruri-v3の微調整に使われたデータを使ってSAEを学習します。

```bash
muscari train-sae \
  --trainer topk \
  --model cl-nagoya/ruri-v3-310m \
  --dataset cl-nagoya/ruri-v3-dataset-ft \
  --subsets auto-wiki-qa-nemotron jaquad jqara jsquad miracl mkqa mr-tydi quiz-no-mori quiz-works \
  --columns "anc=検索クエリ: " "pos=検索文書: " "neg=検索文書: " \
  --checkpoint-every 1000 \
  --out runs/ruri-v3-310m-sae
```

`--columns`には、読み込む列と、その列のテキストにエンコーダが期待するプレフィックスを指定します。統計はプレフィックスごとに出力するので、クエリと文書が分かれます。プレフィックスを使わないエンコーダでは、代わりに列にラベルを付けます。`--columns "anc:query=" "pos:document=" "neg:document="`のように書きます。既定値は潜在65,536個、`k = 16`、10億トークンで、[Shizukuが使うSAEのモデルカード](https://huggingface.co/mahiyama/modernbert-ja-310m-sae)に書かれた値です。最初に試すときは、`--num-latents`と`--total-tokens`を小さくし、`cl-nagoya/ruri-v3-30m`のような小さいモデルを使ってください。

同じ活性をSAELensで学習する場合：

```bash
muscari train-sae --trainer saelens --architecture topk ...
```

コーパスは何でも使えます。1行1テキストのテキストファイルは`text`という列として読み込まれます。

```bash
muscari train-sae \
  --trainer topk \
  --model cl-nagoya/ruri-v3-310m \
  --data-files corpus.txt \
  --columns "text=検索文書: " \
  --out runs/my-corpus-sae
```

書き出したSAEを別のテキストで評価できます。省略したエンコーダ設定は、SAEの`sae_config.json`から読み込みます。

```bash
muscari eval-sae --sae runs/ruri-v3-310m-sae --data-files other.jsonl --columns "text=検索文書: "
```

それぞれの潜在が何に反応しているかを、発火の多い順に一覧にできます。表は端末に、全件の一覧はSAEと同じ場所の`latents.jsonl`に出力します。

```bash
muscari inspect-sae --sae runs/ruri-v3-310m-sae --data-files other.jsonl --columns "text=検索文書: "
```

検索の学習をする前に、凍結したエンコーダとSAEだけで、並べ替え用データセットの候補を並べ替えられます。SAEを経路に入れてもエンコーダがうまく順位付けできるか、SAEの特徴だけでどこまで順位付けできるかが分かります。

```bash
muscari probe-sae --sae runs/ruri-v3-310m-sae --benchmark jqara \
  --query-prefix "検索クエリ: " --passage-prefix "検索文書: "
```

すべてのオプションは`muscari train-sae --help`で確認できます。ダウンロードなしで動く例は[examples](examples/README.ja.md)にあります。

## Python API

```python
from muscari import (
    EncoderActivationSource,
    TopKTrainer,
    TopKTrainerConfig,
    TextColumn,
    iter_texts,
    load_sae,
    repeat_texts,
    save_topk_sae,
    shuffled_batches,
)

rows = [{"query": "...", "passage": "..."}]  # マッピングを返す任意のイテラブル
columns = [
    TextColumn("query", "検索クエリ: "),
    TextColumn("passage", "検索文書: "),
]

source = EncoderActivationSource.from_pretrained("cl-nagoya/ruri-v3-30m")
batches = shuffled_batches(
    source.stream(repeat_texts(lambda: iter_texts(rows, columns))),
    batch_tokens=8192,
    buffer_tokens=524_288,
)

config = TopKTrainerConfig(
    num_latents=16_384,
    k=16,
    total_tokens=50_000_000,
)
trainer = TopKTrainer(config, d_in=source.d_in)
sae = trainer.fit(batches, out_dir="runs/sae")
save_topk_sae("runs/sae", sae, {"base_model": "cl-nagoya/ruri-v3-30m"})

exported = load_sae("runs/sae", device=source.device)
hidden = source.encode_batch(["検索文書: "], ["名古屋の求人"])  # (トークン数, d_in)
features = exported.encode(hidden)  # (トークン数, 潜在数)。非ゼロはトークンあたりk個
```

## 疎検索のヘッド

`muscari.retrieval`は、書き出したSAEをエンコーダに載せ、[Sentence Transformers](https://sbert.net/)の`SparseEncoder`にします。ヘッドの`SAEPooling`は、SPLADEモデルの言語モデルヘッドとプーリングの役割を担います。本文トークンの最終隠れ状態をSAEのエンコーダに通し、発火した潜在に`log1p`をかけ、トークン方向の最大値を取ります。潜在1個が出力の1次元になります。

```python
from muscari.retrieval import build_sparse_encoder

model = build_sparse_encoder(
    "runs/ruri-v3-310m-sae",
    query_prompt="検索クエリ: ",
    document_prompt="検索文書: ",
)
queries = model.encode_query(["名古屋のエンジニア求人"])
documents = model.encode_document(["名古屋市内の開発職を募集しています。", "東京の営業職です。"])
scores = model.similarity(queries, documents)
```

学習には、`sentence_transformers.sparse_encoder`の損失、トレーナー、評価器をそのまま使えます。ヘッドを`trainable=True`で作らない限り、SAEは凍結したままです。

```python
from sentence_transformers.sparse_encoder import SparseEncoderTrainer
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss

loss = SpladeLoss(
    model,
    SparseMultipleNegativesRankingLoss(model),
    document_regularizer_weight=...,  # 要調整。このヘッドで検証した値はまだありません
    query_regularizer_weight=...,
)
trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
trainer.train()
model.save("runs/ruri-v3-310m-sae-retriever")
```

保存したモデルは、muscariをインストールした環境で`SparseEncoder(path, trust_remote_code=True)`として読み込みます。Sentence Transformersは、自身の名前空間の外にあるモジュールクラスすべてにこの指定を求めます。

ほぼすべてのテキストで活性のある潜在は、テキストを区別する役に立ちません。`inspect-sae`の`text_frequency`で見つけ、`build_sparse_encoder(..., head_kwargs={"masked_latents": [...]})`で出力から外せます。

ヘッドは正しさを検証済みです。出力するベクトルは`probe-sae`が採点するベクトルと一致し、勾配は密な計算と一致します。小さなモデルで数ステップ学習すると、エンコーダが更新され、SAEは変わりません。このヘッドで検索モデルを学習したことは、まだありません。

## 出力ファイル

| ファイル | 内容 |
| --- | --- |
| `sae.safetensors` | `encoder.weight (M, d)`、`encoder.bias (M)`、`decoder.weight (d, M)`、`decoder.bias (d)`。JumpReLU型のSAEでは`threshold (M)`も含みます |
| `sae_config.json` | サイズ、活性化の規則、`k`、元のモデル、層、データセット、列とプレフィックス、ホールドアウトの割合、トレーナー、学習トークン数 |
| `metrics.json` | 学習後のホールドアウト評価、辞書の使われ方、トークン数、所要時間 |
| `train_log.jsonl`、`trainer_config.json`、`checkpoint.pt` | `topk`トレーナーのみ |
| `latents.jsonl` | `inspect-sae`が書き出します。潜在1個につき1行で、発火率、テキスト内の位置、よく発火するトークンが入ります |

デコーダのバイアスはエンコーダのバイアスに畳み込んであります。符号化はアフィン変換一つで済みます。`pre = x @ encoder.weight.T + encoder.bias`を計算し、トークンごとにReLU後の上位`k`個を残します。ローカルのパスはファイル名だけを記録します。

## 二つのトレーナー

| | `topk` | `saelens` |
| --- | --- | --- |
| 実装 | このリポジトリ | [SAELens](https://pypi.org/project/sae-lens/)の`SAETrainer`（任意の追加依存） |
| 構造 | Top-K | Top-K、BatchTopK（JumpReLUとして書き出し） |
| 補助損失 | AuxK。`aux_k`を指定でき、残差の分散で正規化します | SAELensの実装。死んだ潜在を`d_in // 2`個使い、正規化しません |
| 死んだ潜在の判定 | トークン数で数えます | ステップ数で数えます |
| 学習率 | 5%の線形ウォームアップ、0までのコサイン減衰 | 線形ウォームアップ、`lr / 100`までのコサイン減衰 |
| デコーダのバイアスの初期値 | 最初のバッチの幾何中央値 | 同じ。`--zero-b-dec`でSAELens本来の0に戻せます |
| チェックポイントと再開 | あり（`--checkpoint-every`、`--resume`） | なし |
| ログ | JSON Lines | SAELens。Weights & Biasesも使えます |

`topk`はこのリポジトリ独自の学習ループです。仕組みはGaoらに従い、既定のハイパーパラメータはmahiyama氏の[Shizuku](https://huggingface.co/mahiyama/sae-splade-ja-310m)が使うSAEのモデルカードの値です。元の学習コードは公開されておらず、このプロジェクトは作者と関係ありません。カードに書かれていること、書かれていないこと、カードの設定で実行する方法は[設計ガイド](docs/ja/design.md#shizukuのsaeのカードの設定で実行する)にまとめています。

## 評価

`metrics.json`と`eval-sae`は、ホールドアウトのトークンで次の指標を出力します。

| 指標 | 意味 |
| --- | --- |
| `fvu` | 説明できなかった分散の割合。再構成の二乗誤差を、平均活性まわりの分散で割った値です |
| `l0` | トークンあたりの発火した潜在の平均個数 |
| `never_fired_frac` | 評価したトークンで一度も発火しなかった潜在の割合 |
| `rare_frac` | 平均的な潜在の10分の1未満しか発火しなかった潜在の割合 |
| `dense_frac` | トークンの10%超で発火した潜在の割合 |
| `median_frequency`、`max_frequency` | 中央の潜在の発火率と、最も頻繁に発火した潜在の発火率 |
| `by_prefix` | プレフィックスごと、つまりクエリと文書を分けた、トークンの比率・`fvu`・`l0`。列にラベルを付けた場合は、ここから下のプレフィックスはラベルに置き換わります |

疎検索のインデックスが持つのはトークンではありません。クエリや文書1件ごとに、トークン方向に最大値でプーリングしたベクトルを1つ持ち、その非ゼロ次元の数だけコストがかかります。`text_level`は、このプーリング後の辞書の性質です。潜在がテキストの中の1トークンでも発火していれば、そのテキストで活性があると数えます。

| 指標 | 意味 |
| --- | --- |
| `texts`、`tokens_per_text` | このプレフィックスで評価したテキストの件数と、平均の長さ |
| `l0`、`l0_median`、`l0_p95` | テキストあたりの活性のある潜在の個数。平均、中央値、95パーセンタイル |
| `never_fired_frac` | どのテキストでも活性のなかった潜在の割合 |
| `dense_frac` | テキストの10%超で活性のあった潜在の割合 |
| `max_document_frequency` | 1つの潜在が活性を持つテキストの割合の最大値 |
| `flops` | プレフィックスの組ごとに、潜在の活性確率の積を全潜在で合計した値。一方の種類のテキスト1件ともう一方の1件を転置インデックスで照合するときの、掛け算の回数の期待値です |

クエリのプレフィックスと文書のプレフィックスの組の`flops`は、[naver/splade](https://github.com/naver/splade)がクエリと文書のインデックスから出力するFLOPSと同じ量で、[Pariaら](https://arxiv.org/abs/2004.05665)に由来します。ここでの値はSAEだけの性質です。検索の学習はまだ何もしておらず、学習すれば値は変わります。

`inspect-sae`は、潜在1個につき1レコードを`latents.jsonl`に書き出します。

| 項目 | 意味 |
| --- | --- |
| `frequency`、`frequency_by_prefix` | 潜在が発火したトークンの割合。全体と、プレフィックスごとのトークンの中での値 |
| `text_frequency` | 1トークンでも発火したテキストの割合 |
| `mean_activation` | 発火したときの値の平均 |
| `first_token_share`、`last_token_share`、`mean_position` | 発火のうち、テキストの最初の本文トークンと最後の本文トークンで起きた割合と、発火位置の相対的な平均（0が先頭、1が末尾） |
| `top_tokens` | よく発火するトークン。それぞれに`share`（この潜在の発火のうち、このトークンで起きた割合）と`token_rate`（このトークンの出現のうち、この潜在が発火した割合）が付きます |

潜在は、意味ではなく、特定のトークン、テキストの種類、テキスト内の位置のどれかに反応していることがあります。これらの項目は、その3つを見分けるためのものです。

### エンコーダ自身のヘッドを通して測る

再構成誤差からは、SAEが落とした情報が重要かどうかは分かりません。そこで、SAEをエンコーダの中に戻し、エンコーダがそれをどう使うかを2通りの方法で測ります。

`probe-sae`は、候補の文書が1行に1件入った並べ替え用データセットを読み、クエリごとに候補を並べ替えます。`--benchmark jqara`は[JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA)のテスト分割を読みます。ほかのデータは、`--dataset`または`--data-files`と、`--query-column`、`--passage-column`、`--label-column`、`--group-column`で指定します。既定では300件のクエリを抜き出し、どの値にもそのクエリについての標準誤差（`ndcg@10_stderr`）を付けます。

| 結果 | 意味 |
| --- | --- |
| `dense.original` | エンコーダの平均プーリングの埋め込みのコサイン類似度で並べ替えたときのnDCG@10とMRR@10 |
| `dense.spliced` | 本文トークンの状態をすべてSAEの再構成に置き換えてから同じことをした値 |
| `dense.empty` | 本文トークンの状態をすべてSAEのデコーダのバイアスに置き換えた値。空の符号に対してSAEが返すものです |
| `dense.recovered` | `(spliced - empty) / (original - empty)`。エンコーダの順位付けの質のうち、SAEを通しても残る割合です |
| `dense.cosine` | 同じテキストの、元の埋め込みと置き換え後の埋め込みのコサイン類似度。平均、5パーセンタイル、最小値 |
| `sparse.plain` | プーリングしたSAEの特徴の内積で並べ替えた値。`log1p`をかけ、本文トークン方向の最大値を取ります |
| `sparse.idf` | 同じ内積で、各潜在に候補の文書の中での逆文書頻度の重みを付けた値 |
| `sparse.l0` | クエリ1件、文書1件あたりの活性のある潜在の個数 |

`dense`の行は、ruri-v3のように平均プーリングの埋め込みを出すよう学習されたエンコーダで意味を持ちます。`sparse`の行は、SAEを語彙にした疎検索モデルの学習前の状態にあたります。出発点であり、成果ではありません。

言語モデルのヘッドが残っているエンコーダでは、`eval-sae --mlm-texts 100`で`mlm`が加わります。マスク言語モデルの損失を、エンコーダをそのまま使った場合（`loss_clean`）、本文トークンの最終状態をSAEの再構成に置き換えた場合（`loss_spliced`）、その平均に置き換えた場合（`loss_mean`）で測り、`recovered = (loss_mean - loss_spliced) / (loss_mean - loss_clean)`を出力します。ShizukuのSAEのカードが報告しているのはこの量です。カードにはマスクする割合が書かれていません。ここでの既定値は15%で、使った値は結果と一緒に記録します。

学習後の`metrics.json`には、次の値も入ります。

| 指標 | 意味 |
| --- | --- |
| `train_unused_frac` | 学習の最後の1,000万トークンで一度も発火しなかった潜在の割合（学習がそれより短ければ全体） |
| `tokens_encoded_by_prefix` | 学習に使ったトークンのうち、クエリ由来と文書由来の数 |
| `epochs_started` | 重複除去後のテキストを何周目まで読み始めたか |
| `encoder_seconds`、`sae_seconds`、`wall_seconds` | エンコーダ、SAEの更新（`topk`トレーナー）、全体にかかった時間 |

ホールドアウトに回すテキストは、テキスト自身のハッシュで決めます。多くの行に現れる文書が学習側と評価側に分かれることはありません。評価するトークン数の既定値は潜在1個あたり32トークンで、最低でも20万です。これより少ないと、まれにしか発火しない潜在に発火する機会がなく、`never_fired_frac`が大きく出ます。`eval-sae`は、SAEの学習に使ったデータを渡された場合だけホールドアウト分を評価し、それ以外のデータではすべてを評価します。

プローブが測るのは、何も学習していない状態での順位付けです。学習した検索モデルの性能を測る指標はありません。

## 検証状況

| 環境 | 実行した内容 | 結果 |
| --- | --- | --- |
| GitHub Actions、Ubuntu 24.04 | Python 3.12でロック済みの`mise run check`。Python 3.11で対応範囲の最も古い版（PyTorch 2.6、Transformers 4.48、Datasets 3.0、safetensors 0.4.4）のテスト。Python 3.13で最新の版とSAELensを入れたテスト | 成功 |
| macOS 26.6、Apple M5 Pro、Python 3.12、PyTorch 2.15 | `mise run check`。MPSでのハードウェアのテストとBetterleaksを含みます | 成功 |
| 同じMac、MPS | `cl-nagoya/ruri-v3-30m`で500万〜3,000万トークンの学習（両方のトレーナー）と、ステップのベンチマーク | 最後まで動作。数値は[設計ガイド](docs/ja/design.md#実際のエンコーダでの試走)を参照 |
| Google Colab、NVIDIA L4、Python 3.13、PyTorch 2.14 | テスト。CUDAでの損失と勾配をCPUと比較。bfloat16のautocast。ステップのベンチマーク。500万トークンの学習をfloat32とbfloat16で実行 | 成功。数値は設計ガイドを参照 |
| Linux x86_64 CPU、Python 3.13、PyTorch 2.15 | 整形、lint、mypy、テスト、パッケージビルド、公開内容の監査、ステップのベンチマーク | 成功 |

未実行：潜在65,536個・10億トークンの本来の規模での学習、検索性能の測定。自動テストはランダムに初期化した2層のBERTを使います。確認しているのはインターフェースと数値計算で、SAEの品質ではありません。ruri-v3-30mでの学習は、小さな辞書で、それぞれシード一つです。

## 制限

- 1プロセス、1デバイスで動きます。複数GPUへの分割には対応していません。
- `--resume`はSAE、オプティマイザ、死んだ潜在の状態を正確に復元します。データストリームは、符号化済みのテキストの続きから、新しいシャッフルのシードで再開します。チェックポイント時点でシャッフルバッファに残っていた活性は使われません。
- データセットは全体を読み込み、行単位でシャッフルします。ストリーミングには対応していません。
- 活性はエンコーダの出力をそのまま使います。Gaoらが行っているトークンごとの正規化は行いません。
- SAELensバックエンドは再開できません。補助損失の細部もここからは変更できません。
- `probe-sae`と`--mlm-texts`は、`last_hidden_state`で学習したSAEでだけ使えます。

## 今後の予定

- 潜在65,536個・10億トークンでの基準となる学習と、再構成・プローブの測定結果。
- 検索ヘッドを使った検索学習と、語彙型のSPLADEモデルとの比較。
- ストリーミングデータセットへの対応。

## 謝辞

- Gaoら、[Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)：Top-K SAE、AuxK、初期化。細部の確認のために[参照実装](https://github.com/openai/sparse_autoencoder)を読みました。疎な逆伝播もこの実装に着想を得ています。
- mahiyama氏の[Shizuku](https://huggingface.co/mahiyama/sae-splade-ja-310m)と、`topk`トレーナーが従う[SAEのモデルカード](https://huggingface.co/mahiyama/modernbert-ja-310m-sae)。
- [From Tokens to Concepts: Leveraging SAE for SPLADE](https://arxiv.org/abs/2604.21511)（SAE-SPLADE）と[Learning Retrieval Models with Sparse Autoencoders](https://arxiv.org/abs/2603.13277)（SPLARE）：このツールが対象とする検索の枠組み。
- [SAELens](https://pypi.org/project/sae-lens/)：二つ目の学習バックエンド。
- [ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m)とその[微調整データセット](https://huggingface.co/datasets/cl-nagoya/ruri-v3-dataset-ft)：説明の例として使っています。

## ライセンス

Apache-2.0です。[LICENSE](LICENSE)と[NOTICE.md](NOTICE.md)を参照してください。
