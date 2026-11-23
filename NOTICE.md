# Licensing and third-party references

muscari

Copyright 2026 The muscari contributors.

muscari is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the terms. This notice provides attribution and does not modify the license.

PyTorch, Transformers, Datasets, safetensors and the optional SAELens backend are installed as separate dependencies. They retain their own licenses and notices; their source distributions are not bundled in muscari's wheel or source archive.

The Top-K sparse autoencoder follows Gao et al., "Scaling and evaluating sparse autoencoders". The [reference implementation](https://github.com/openai/sparse_autoencoder/tree/4965b941e9eb590b00b253a2c406db1e1b193942) (MIT License) was read to check details; none of its code is bundled in muscari. The `topk` trainer follows the recipe described on the model card of [mahiyama/modernbert-ja-310m-sae](https://huggingface.co/mahiyama/modernbert-ja-310m-sae); the training code of those models is not public, and muscari is an independent implementation that is not affiliated with their author. The research references are listed in the [design guide](https://github.com/tkhshtsh0917/muscari/blob/main/docs/design.md#research-background). Preserve applicable upstream licenses, notices and attribution when redistributing third-party material.

Models and datasets named in the documentation, such as ruri-v3 and its fine-tuning dataset, are not distributed with muscari. Check their licenses before using them or publishing an SAE trained on them.

## 日本語

muscariはApache License, Version 2.0で提供します。条件は[LICENSE](LICENSE)を参照してください。この文書は出典を説明するもので、ライセンスを変更するものではありません。

PyTorch、Transformers、Datasets、safetensors、任意で使うSAELensは別の依存ライブラリとしてインストールされ、それぞれのライセンス・表示が適用されます。これらのソース配布物はmuscariのwheel・source archiveへ同梱していません。

Top-K SAEはGaoらの「Scaling and evaluating sparse autoencoders」に従っています。細部の確認のために[参照実装](https://github.com/openai/sparse_autoencoder/tree/4965b941e9eb590b00b253a2c406db1e1b193942)（MIT License）を読みましたが、そのコードはmuscariへ同梱していません。`topk`トレーナーは[mahiyama/modernbert-ja-310m-sae](https://huggingface.co/mahiyama/modernbert-ja-310m-sae)のモデルカードに書かれた学習設定に従っています。これらのモデルの学習コードは公開されておらず、muscariは作者と関係のない独立した実装です。関連研究は[設計ガイド](https://github.com/tkhshtsh0917/muscari/blob/main/docs/ja/design.md#関連研究)に記載しています。第三者の素材を再配布する場合は、その素材に適用されるライセンス・表示・出典を保持してください。

ruri-v3やその微調整データセットなど、文書に登場するモデルとデータセットはmuscariに含まれません。利用する前、また学習したSAEを公開する前に、それぞれのライセンスを確認してください。
