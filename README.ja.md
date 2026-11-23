# muscari

[English](README.md)

テキストエンコーダ向けのスパースオートエンコーダ（SAE）です。エンコーダの隠れ状態でSAEを学習し、SAE型の学習型疎検索の語彙として使います。

muscariは、[ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m)のような凍結したテキストエンコーダにテキストを通し、本文トークンごとの隠れ状態でTop-K SAEを学習します。得られるのは数万本の方向からなる辞書です。検索モデルは、トークナイザの語彙の代わりにこの方向の上でクエリと文書を採点できます。SAE-SPLADEやSPLAREの考え方です。

現在は初期版（`0.1.0.dev0`）で、SAEの学習・書き出し・評価までを扱います。SAEを使った検索学習はまだ含まれていません。[今後の予定](#今後の予定)を参照してください。

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
```

Python 3.11以降、PyTorch 2.6以降、Transformers 4.48以降が必要です。開発環境は[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)を参照してください。

## クイックスタート

ruri-v3の最終層で、ruri-v3の微調整に使われたデータを使ってSAEを学習します。

```bash
muscari train-sae \
  --trainer shizuku \
  --model cl-nagoya/ruri-v3-310m \
  --dataset cl-nagoya/ruri-v3-dataset-ft \
  --subsets auto-wiki-qa-nemotron jaquad jqara jsquad miracl mkqa mr-tydi quiz-no-mori quiz-works \
  --columns "anc=検索クエリ: " "pos=検索文書: " "neg=検索文書: " \
  --checkpoint-every 1000 \
  --out runs/ruri-v3-310m-sae
```

`--columns`には、読み込む列と、その列のテキストにエンコーダが期待するプレフィックスを指定します。既定値は潜在65,536個、`k = 16`、10億トークンで、[Shizukuが使うSAEのモデルカード](https://huggingface.co/mahiyama/modernbert-ja-310m-sae)に書かれた値です。最初に試すときは、`--num-latents`と`--total-tokens`を小さくし、`cl-nagoya/ruri-v3-30m`のような小さいモデルを使ってください。

同じ活性をSAELensで学習する場合：

```bash
muscari train-sae --trainer saelens --architecture topk ...
```

コーパスは何でも使えます。1行1テキストのテキストファイルは`text`という列として読み込まれます。

```bash
muscari train-sae \
  --trainer shizuku \
  --model cl-nagoya/ruri-v3-310m \
  --data-files corpus.txt \
  --columns "text=検索文書: " \
  --out runs/my-corpus-sae
```

書き出したSAEを別のテキストで評価できます。省略したエンコーダ設定は、SAEの`sae_config.json`から読み込みます。

```bash
muscari eval-sae --sae runs/ruri-v3-310m-sae --data-files other.jsonl --columns "text=検索文書: "
```

すべてのオプションは`muscari train-sae --help`で確認できます。ダウンロードなしで動く例は[examples](examples/README.ja.md)にあります。

## Python API

```python
from muscari import (
    EncoderActivationSource,
    ShizukuTrainer,
    ShizukuTrainerConfig,
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

config = ShizukuTrainerConfig(
    num_latents=16_384,
    k=16,
    total_tokens=50_000_000,
)
trainer = ShizukuTrainer(config, d_in=source.d_in)
sae = trainer.fit(batches, out_dir="runs/sae")
save_topk_sae("runs/sae", sae, {"base_model": "cl-nagoya/ruri-v3-30m"})

exported = load_sae("runs/sae", device=source.device)
hidden = source.encode_batch(["検索文書: "], ["名古屋の求人"])  # (トークン数, d_in)
features = exported.encode(hidden)  # (トークン数, 潜在数)。非ゼロはトークンあたりk個
```

## 出力ファイル

| ファイル | 内容 |
| --- | --- |
| `sae.safetensors` | `encoder.weight (M, d)`、`encoder.bias (M)`、`decoder.weight (d, M)`、`decoder.bias (d)`。JumpReLU型のSAEでは`threshold (M)`も含みます |
| `sae_config.json` | サイズ、活性化の規則、`k`、元のモデル、層、データセット、列とプレフィックス、ホールドアウトの割合、トレーナー、学習トークン数 |
| `metrics.json` | 学習後のホールドアウト評価 |
| `train_log.jsonl`、`trainer_config.json`、`checkpoint.pt` | `shizuku`トレーナーのみ |

デコーダのバイアスはエンコーダのバイアスに畳み込んであります。符号化はアフィン変換一つで済みます。`pre = x @ encoder.weight.T + encoder.bias`を計算し、トークンごとにReLU後の上位`k`個を残します。ローカルのパスはファイル名だけを記録します。

## 二つのトレーナー

| | `shizuku` | `saelens` |
| --- | --- | --- |
| 実装 | このリポジトリ | [SAELens](https://pypi.org/project/sae-lens/)の`SAETrainer`（任意の追加依存） |
| 構造 | Top-K | Top-K、BatchTopK（JumpReLUとして書き出し） |
| 補助損失 | AuxK。`aux_k`を指定でき、残差の分散で正規化します | SAELensの実装。死んだ潜在を`d_in // 2`個使い、正規化しません |
| 死んだ潜在の判定 | トークン数で数えます | ステップ数で数えます |
| 学習率 | 5%の線形ウォームアップ、0までのコサイン減衰 | 線形ウォームアップ、`lr / 100`までのコサイン減衰 |
| チェックポイントと再開 | あり（`--checkpoint-every`、`--resume`） | なし |
| ログ | JSON Lines | SAELens。Weights & Biasesも使えます |

`shizuku`は、mahiyama氏の[Shizuku](https://huggingface.co/mahiyama/sae-splade-ja-310m)が使うSAEのモデルカードに書かれた学習設定を、独立に実装したものです。元の学習コードは公開されておらず、このプロジェクトは作者と関係ありません。詳細とGaoらとの違いは[設計ガイド](docs/ja/design.md)にまとめています。

## 評価

`metrics.json`と`eval-sae`は次の指標を出力します。

| 指標 | 意味 |
| --- | --- |
| `fvu` | 説明できなかった分散の割合。再構成の二乗誤差を、平均活性まわりの分散で割った値です |
| `l0` | トークンあたりの発火した潜在の平均個数 |
| `never_fired_frac` | 評価したトークンで一度も発火しなかった潜在の割合 |
| `dense_frac` | トークンの10%超で発火した潜在の割合 |
| `max_frequency` | 最も頻繁に発火した潜在の発火率 |

ホールドアウトに回すテキストは、テキスト自身のハッシュで決めます。多くの行に現れる文書が学習側と評価側に分かれることはありません。これらの指標は再構成と辞書の使われ方を表すもので、検索性能を測るものではありません。

## 検証状況

| 環境 | 実行した内容 | 結果 |
| --- | --- | --- |
| Linux x86_64 CPU、Python 3.13、PyTorch 2.15 | 整形、lint、mypy、テスト、パッケージビルド、公開内容の監査 | 成功 |
| Linux x86_64 CPU、Python 3.13、PyTorch 2.6、Transformers 4.48、Datasets 3.0、safetensors 0.4.4 | SAELensなしのテスト | 成功 |

未実行：Apple MPS、CUDA（CUDA上のbfloat16 autocastを含む）、実際のエンコーダでの学習。テストはすべて、ランダムに初期化した2層のBERTを使います。確認しているのはインターフェースと数値計算で、SAEの品質ではありません。

## 制限

- 1プロセス、1デバイスで動きます。複数GPUへの分割には対応していません。
- `--resume`はSAE、オプティマイザ、死んだ潜在の状態を正確に復元します。データストリームは、符号化済みのテキストの続きから、新しいシャッフルのシードで再開します。チェックポイント時点でシャッフルバッファに残っていた活性は使われません。
- データセットは全体を読み込み、行単位でシャッフルします。ストリーミングには対応していません。
- 活性はエンコーダの出力をそのまま使います。Gaoらが行っているトークンごとの正規化は行いません。
- SAELensバックエンドは再開できません。補助損失の細部もここからは変更できません。

## 今後の予定

- 書き出したSAEをエンコーダに載せ、疎検索のヘッドとして使うSentence Transformersモジュールと、それを使った検索学習。
- ruri-v3での基準となる学習と、再構成・検索の測定結果。
- ストリーミングデータセットへの対応。

## 謝辞

- Gaoら、[Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)：Top-K SAE、AuxK、初期化。細部の確認のために[参照実装](https://github.com/openai/sparse_autoencoder)を読みました。疎な逆伝播もこの実装に着想を得ています。
- mahiyama氏の[Shizuku](https://huggingface.co/mahiyama/sae-splade-ja-310m)と、`shizuku`トレーナーが従う[SAEのモデルカード](https://huggingface.co/mahiyama/modernbert-ja-310m-sae)。
- [From Tokens to Concepts: Leveraging SAE for SPLADE](https://arxiv.org/abs/2604.21511)（SAE-SPLADE）と[Learning Retrieval Models with Sparse Autoencoders](https://arxiv.org/abs/2603.13277)（SPLARE）：このツールが対象とする検索の枠組み。
- [SAELens](https://pypi.org/project/sae-lens/)：二つ目の学習バックエンド。
- [ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m)とその[微調整データセット](https://huggingface.co/datasets/cl-nagoya/ruri-v3-dataset-ft)：説明の例として使っています。

## ライセンス

Apache-2.0です。[LICENSE](LICENSE)と[NOTICE.md](NOTICE.md)を参照してください。
