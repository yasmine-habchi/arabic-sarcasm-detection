# Arabic Sarcasm Detection with Pragmatic Features

Master's thesis project (Faculty of Sciences of Bizerte, University of Carthage).
It studies whether 18 explicit pragmatic features improve Arabic sarcasm detection when they are combined with frozen contextual embeddings (MARBERT, AraBERT, CamelBERT) in a SAtt-BLSTM-ConvNet classifier.

## Approach

1. **Preprocessing**: Arabic text cleaning (URLs, mentions, hashtags, hamza and ligature normalisation, diacritics removal).
2. **Pragmatic features (18)**: sentiment scores (CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment), intensifiers, irony markers, negation, hyperbole, punctuation, Latin ratio, a discriminative vocabulary, hashtags, emoji count, interjections, quotations, repeated characters and tweet length.
3. **Embeddings**: full sequences (64 x 768) from MARBERT, AraBERT and CamelBERT, kept frozen.
4. **Classifier**: SAtt-BLSTM-ConvNet (1D convolution, BiLSTM, soft attention), concatenated with the 18 features. It is inspired by Son et al. (2019) with a different module order.
5. **Evaluation**: 5 random seeds, macro F1 and MCC, one-sided paired Wilcoxon test (with vs without features), and an ablation study (leave-one-feature-out, by group, cumulative).

## Results (macro F1, mean over 5 seeds)

| Dataset | Model | Without features | With features | Wilcoxon p |
|---|---|---|---|---|
| ArSarcasm-v2 | MARBERT | 0.7047 | 0.7131 | 0.0938 |
| ArSarcasm-v2 | AraBERT | 0.6552 | 0.6752 | 0.0625 |
| ArSarcasm-v2 | CamelBERT | 0.6896 | 0.6941 | 0.1562 |
| ArSarcaMoji | MARBERT | 0.6965 | 0.7127 | 0.0312 |
| ArSarcaMoji | AraBERT | 0.6589 | 0.6816 | 0.0312 |
| ArSarcaMoji | CamelBERT | 0.6733 | 0.6783 | 0.4062 |

On ArSarcaMoji, the improvement is statistically significant for MARBERT and AraBERT, and the ablation shows that it comes mainly from the emoji count. On ArSarcasm-v2, the improvements are consistent but not significant at the 5% level.

## A note on emojis

A first protocol gave the embedding models the text with emojis and reached a macro F1 of about 0.99 on ArSarcaMoji. Some frequent emojis almost perfectly predict the label, so the score did not measure sarcasm detection. In the final protocol, embeddings are computed on the text **without emojis** (`Plain_Text` column). Emojis still reach the classifier, but only through the pragmatic features: mainly the emoji count, and also the sentiment scores, which are computed on the text with emojis. All ArSarcaMoji results above use this protocol.

## Repository structure

    ArSarcasm-v2/
      02_pragmatic_features.ipynb          18 pragmatic features
      08_marbert_embeddings.ipynb          MARBERT sequence embeddings (64 x 768)
      09_sattblstmconvnet_final.ipynb      MARBERT: runs, Wilcoxon, ablation
      10_arabert_arsarcasmv2.ipynb         AraBERT (embeddings + runs)
      11_camelbert_arsarcasmv2.ipynb       CamelBERT (embeddings + runs)
    ArSarcaMoji/
      01_exploration.ipynb                 data loading, binary label, split
      02_dedup_and_features.ipynb          duplicate removal, features, emoji analysis
      09_marbert_correction.ipynb          MARBERT, text without emojis
      10_arabert_correction_plaintext.ipynb
      11_camelbert_correction.ipynb

## Data

The datasets and the embedding files are **not included**.
- ArSarcasm-v2: Abu Farha et al., 2021.
- ArSarcaMoji: `arbml/ArSarcasMoji` on Hugging Face (use the `Plain_Text` column for embeddings).

The notebooks expect the data in `data/` (at the repository root) and write their outputs in `outputs*/` folders.

## How to run

Run the notebooks in this order, because each one reloads the files written by the previous ones.

**ArSarcasm-v2**
1. `02_pragmatic_features`
2. `08_marbert_embeddings` (slow: extracts the MARBERT embeddings)
3. `09_sattblstmconvnet_final`
4. `10_arabert_arsarcasmv2` and `11_camelbert_arsarcasmv2` (each extracts its own embeddings)

**ArSarcaMoji**
1. `01_exploration`
2. `02_dedup_and_features`
3. `09_marbert_correction`
4. `10_arabert_correction_plaintext` (it also reloads the MARBERT outputs of step 3 for comparison plots)
5. `11_camelbert_correction`

## Limitations

- The architecture is adapted from the original papers, not replicated.
- With 5 seeds, the smallest possible Wilcoxon p-value is 0.0312.
- The train/validation/test split is fixed, so its variability is not measured.
- The ablation sets features to zero at inference time without retraining.
- Ablation was run only for the configurations with a significant effect.
- On ArSarcaMoji, 6 of the 18 features are constant and 2 are almost constant, so only about 10 features carry information on this dataset.
- The sentiment scores are computed on the text with emojis, so part of the emoji signal also enters through them, not only through the emoji count.

## Installation

    pip install -r requirements.txt