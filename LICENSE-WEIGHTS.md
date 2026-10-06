# Licences of the released weights

The code in this repository is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). The model
weights on the Hugging Face Hub ([`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET)) are
licensed separately:

| Weights | Licence |
|---|---|
| GeoSET generalist transformer (`generalist/transformer`), SAR encoder (`generalist/sar_encoder`) and LoRA adapters (`lora/<dataset>`) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| Autoencoder (`generalist/vae`): the FLUX.2-klein-base-4B autoencoder of Black Forest Labs, converted to float32 | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Comparison methods (`baselines/<dataset>/<method>`) | the terms of the code each was trained with; see [MODEL_ZOO.md](MODEL_ZOO.md#licences) and `baselines/licenses/` on the Hub |

**GeoSET weights, CC BY-NC-SA 4.0.** You may share and adapt them for non-commercial purposes, with
attribution (cite the paper, see the [README](README.md)), and adaptations must be distributed under the same
licence. The generalist transformer is initialised from FLUX.2-klein-base-4B and the SAR encoder from its
autoencoder encoder, both Apache-2.0 (see [NOTICE](NOTICE)).

**Datasets.** No training or evaluation data is redistributed. The weights were trained on the datasets
described in [docs/CORPUS.md](docs/CORPUS.md) and [docs/DATASETS.md](docs/DATASETS.md); the terms of those
datasets remain with their owners.
