# muscariへの貢献

[English](CONTRIBUTING.md)

## プロジェクト設定

| 項目 | 正とする場所 |
| --- | --- |
| 目的・成果物 | テキストエンコーダの隠れ状態でSAEを学習・書き出し・評価するPythonライブラリと`muscari`コマンド。[設計](docs/ja/design.md)を参照 |
| 対象環境 | 1プロセス・1デバイス（CPU、CUDA、Apple MPS）。どの環境で何を実行したかは[検証状況](README.ja.md#検証状況)を参照 |
| Python | 開発用の版は`.python-version`、対応する最小の版は`pyproject.toml` |
| ツール | miseが`mise.toml`でuvとBetterleaksの版を固定 |
| 依存関係 | 実行時の範囲と`saelens`の追加依存は`pyproject.toml`。開発ツールは`dev`グループ |
| 再現性 | `uv.lock`が開発とCIの主環境を固定 |
| 品質 | Ruffの整形とlint、mypy、pytest、パッケージビルド、公開内容とライセンスの監査、Betterleaks |
| 文書 | 英語と日本語のREADME、設計ガイド、例のガイド、この文書 |
| 配布 | ソースリポジトリのみ。パッケージ公開のジョブは設定していません |
| ライセンス | Apache-2.0。LICENSEとNOTICEをwheelとsdistに含めます |

パッケージのバージョンは`project.version`で定義します。公開している`__version__`はこれと一致させます。書き出し形式、トークンの選択規則、損失、既定値を変える場合は、対応する文書と、振る舞いを確かめるテストも変更します。

## セットアップと検証

[mise](https://mise.jdx.dev/getting-started.html)をインストールし、チェックアウトした設定を確認してから実行します。

```bash
git clone https://github.com/tkhshtsh0917/muscari.git
cd muscari

mise trust
mise install
mise run setup
mise run check
```

`setup`は、ロックされた依存関係、`saelens`の追加依存、`dev`グループをインストールします。`check`は整形、lint、mypy、テスト、パッケージビルド、監査を`uv run --frozen --no-sync`で実行するので、依存関係の解決や更新は起こりません。依存関係を意図して変える場合は、`pyproject.toml`を編集し、`mise exec -- uv lock`を実行して差分を確認してから、setupとcheckを実行します。`mise run format`は整形を適用します。

CIの主ジョブは、同じロック済みのsetupとcheckを実行します。別の互換性ジョブでは、Python 3.11で対応範囲の最も古い版（SAELensなし）を、Python 3.13で最新の版を解決し、インストールされた版を記録してテストを実行します。ActionsはコミットSHAで固定しています。SHAとバージョンのコメントは一緒に更新してください。

mise環境は`HF_HUB_OFFLINE=1`を設定します。どの検査もモデルやデータセットをダウンロードしません。Hugging Face Hubが必要な実行では、mise環境の内側で変数を設定してください。

```bash
mise exec -- env HF_HUB_OFFLINE=0 uv run --frozen --no-sync muscari train-sae ...
```

`mise exec`の前に`HF_HUB_OFFLINE=0`を書いても効きません。miseがその後で`mise.toml`の値を適用するためです。

## テスト

`tests/`直下のテストは、ランダムに初期化した2層のBERTと合成した活性でライブラリを確認します。確認しているのは数値計算とインターフェースです。疎な逆伝播を密な自動微分・数値微分と比較し、チェックポイントからの再開が正確であること、書き出しがSAELens自身の推論モデルと一致すること、コマンドラインが最初から最後まで動くことを確かめます。`tests/tooling/`は開発用の監査を確認します。SAELensのテストは、追加依存がない場合はスキップされます。

PyTorchがCUDAまたはApple MPSを使えるマシンでは、ハードウェアのテストも実行してください。

```bash
mise exec -- uv run --frozen --no-sync pytest -q -m accelerator -rs
```

損失、勾配、活性、評価の統計をCPUと比較し、そのデバイスで小さなSAEを学習します。CUDAではbfloat16のautocastも確認します。どちらのデバイスもない環境ではスキップされます。ホストされたCIもこれに当たります。どのグループを実行し、どれがスキップされたかを報告してください。

## オフラインの例とベンチマーク

`mise run smoke`は、両方のトレーナーで小さなSAEを学習・評価します。ダウンロードは不要です。標準のcheckがテスト経由で同じ例を実行しているので、実行は任意です。

`benchmarks/sae_step.py`は、疎な逆伝播、または密な自動微分での最適化1ステップの時間を測ります。

```bash
mise exec -- uv run --frozen --no-sync python benchmarks/sae_step.py --mode sparse
mise exec -- uv run --frozen --no-sync python benchmarks/sae_step.py --mode dense --batch-tokens 2048
```

ランダムな活性を使うので、意味があるのは時間とメモリだけです。数値を公開するときは、デバイス、スレッド数、引数、ライブラリの版を一緒に記録し、実行していない条件を明記してください。

## レビューと読みやすいコード

`src/muscari`、`examples`、`benchmarks`、`tests`、`scripts`の責務を分けてください。入力の検証、計算、状態の変更の間には意味のある空行を入れます。複数行にわたる呼び出しでは、1行に1引数を書き、末尾にカンマを付けます。短く単純な呼び出しは1行のままにします。docstringには、責務、形状、失敗条件、数値上の制約を書きます。

修正には、修正前なら失敗するテストを付けます。密な参照実装や、答えのわかる手作りのSAEなど、独立したオラクルと比較するのが理想です。関係する境界値を確認します。ただし、同じ性質をパラメータの全組み合わせで繰り返す必要はありません。ランダムなエンコーダでのテストは、実際のエンコーダでのSAEの品質を保証しません。失敗を隠すために検査を全体で無効にしないでください。

英語と日本語の文書は一緒に更新します。公開ファイルは、手元のメモなしで内容が完結している必要があります。個人のパス、非公開のリンク、トークン、鍵を含めてはいけません。エージェント向けの規則は[AGENTS.md](AGENTS.md)にあります。

## ビルド、ライセンス、秘密情報の検査

`mise run build`は、`dist`にwheelとsdistを一つずつ作ります。`secrets`タスクは、まず公開内容とライセンスの監査を行い、次にBetterleaksを実行します。`check`はこのタスクを1回呼び出します。`mise run audit`は監査だけを実行します。

監査が確認するのは、ライセンスのメタデータ、LICENSEとNOTICEがバイト単位で一致すること、型付けのマーカー、両言語の文書とその相対リンク、sdistの各ファイルが公開ソースの一覧かビルド時に生成される所定のメタデータに属することです。PEP 639のメタデータのためにSetuptools 77.0.3以降が必要です。パッケージのバージョンを変えたときは、`dist`の古い成果物を削除してください。

Betterleaksは固定した版の組み込みルールを使い、公開ファイル、展開した成果物、手元にあるすべてのGit参照を検査します。プロバイダへの問い合わせによる検証は無効です。検出があった場合と検査自体が失敗した場合は、タスクが失敗します。伏せ字にしたレポートは、Gitの管理外の`build/security/`に残ります。浅いクローンでは履歴の検査が失敗するので、足りない履歴を取得してからやり直してください。パターンによる検出は、秘密情報が存在しないことを保証しません。

依存ライブラリにはそれぞれのライセンスが適用されます。第三者の素材を取り込む場合は、必要な表示を保持してください。パッケージを公開する前に、ホストされたCIの結果を確認し、成果物とライセンスを点検し、メンテナの公開手順に従ってください。
