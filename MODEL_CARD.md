# scentvec 1.0.0 model card

## Summary

| | |
|---|---|
| Input | a set of fragrance notes, each optionally tagged top / middle / base |
| Output | one 128-d unit vector; cosine similarity compares perfumes |
| Extra outputs | 21 accord probabilities, 7 families, 15 subfamilies (prototype heads in the same space) |
| Size | 491,055 parameters, 2.3 MB bundle, CPU inference |
| Training | 15 epochs, about 10 minutes on a 4-core laptop CPU, seed 7, checkpoint from epoch 13 |
| Catalog | 44,658 perfumes, 1,888 brands |

## Architecture

Each note becomes the sum of four inputs:

1. a learned embedding for the note itself (notes seen at least 5 times in training),
2. learned embeddings of the words in its name (so "bulgarian rose" shares with "rose"),
3. a 184-d odor-descriptor profile from perfumery chemistry databases, projected,
4. its pyramid position (top, middle, base or unspecified).

One self-attention layer mixes the notes, mean pooling and attention pooling
are concatenated, and a projection plus L2 normalisation gives the embedding.

Training loss: a contrastive loss (temperature 0.07) that pulls together two
views of the same perfume, plus accord and family classification losses on
cosine prototype heads. The two views are either the same perfume as listed by
two independent databases, or two random note-dropout and position-dropout
augmentations of one listing.

## Data

| Source | Used for | Terms |
|---|---|---|
| Parfumo snapshot, TidyTuesday 2024-12-10 (`parfumo_data_clean.csv`) | note pyramids, voted accords, ratings | scraped community data; no clear licence for reuse |
| Hugging Face `doevent/perfume` | flat note lists, family and subfamily, gender | dataset card is tagged MIT; where the rows came from is not verified |
| pyrfume-data: GoodScents, Leffingwell | odor descriptors per molecule (chemistry prior) | research use |
| FlavorGraph nodes and edges | ingredient to molecule links (chemistry prior) | Apache-2.0 |

None of these tables is stored in the repository; `scripts/fetch_raw_data.py`
downloads them from pinned revisions of their original hosts.

After cleaning: 81,138 records, 52,985 with notes (26,666 Parfumo, 26,319
doevent), 2,575 distinct canonical notes, 6,931 perfumes matched across both
databases (same brand and name, release years within one). Splits are by perfume: train 42,445 / val
5,337 / test 5,203 records, with 5,563 / 669 / 699 cross-database pairs.

Cleaning that matters:

- **Fabricated rows.** The Parfumo snapshot contains rows whose note lists were
  machine-generated (they end in implausible notes such as "burnt electronics"
  or nonsense words). 4,496 of 59,315 rows were removed using 119 marker notes.
  Flagged rows agreed with their doevent twin at note Jaccard 0.096, against
  0.594 for unflagged rows. An estimated 500 fabricated rows without a marker
  remain (about 2%), and a few genuine perfumes with unusual notes may have been
  removed.
- **Note canonicalisation.** Spelling variants, plurals and aliases are merged
  ("cedar" to "cedarwood", "tonka" to "tonka bean", and so on).

Chemistry prior coverage of note occurrences: 82.3% direct descriptor match,
8.3% by word, 1.5% by molecule name, 1.1% by ingredient, 6.8% none.

## Evaluation

There is no public ground truth for "these two perfumes smell alike", so the
main test is **cross-database retrieval**: given a perfume's note list from one
database, find the same perfume in the other database, where the notes were
written by different people. A model that only memorises exact note strings does
poorly; a model that understands which notes are interchangeable does well.
All perfumes in the test split were unseen during training.

Baselines: Jaccard overlap of note sets, TF-IDF over notes, TF-IDF over note
words, and LSA (128-d SVD of the TF-IDF matrix). For label prediction: one-vs-rest
logistic regression on a bag of notes.

### Retrieval, test split (MRR, higher is better)

| Task | scentvec | Best baseline | Difference (95% bootstrap interval) | Recall@1 |
|---|---|---|---|---|
| Parfumo to doevent, 691 queries, gallery 2,566 | 0.872 | 0.849 (lsa) | +0.023 (+0.009 to +0.036) | 0.838 vs 0.805 |
| doevent to Parfumo, 699 queries, gallery 2,637 | 0.856 | 0.826 (lsa) | +0.030 (+0.015 to +0.046) | 0.810 vs 0.775 |
| Parfumo to doevent, full gallery 26,319 | 0.779 | 0.739 (lsa) | +0.039 (+0.024 to +0.055) | 0.721 vs 0.676 |
| doevent to Parfumo, full gallery 26,666 | 0.758 | 0.716 (lsa) | +0.042 (+0.024 to +0.059) | 0.685 vs 0.637 |

The gain over the strongest baseline is real but modest: two to four points of
MRR. The bootstrap interval covers query sampling only. The model was trained
with a single seed, so seed-to-seed variation is not measured.

### Label prediction from notes, test split

| | scentvec | Logistic regression |
|---|---|---|
| Accord mAP | 0.781 | 0.782 |
| Family accuracy | 0.639 | 0.640 |
| Subfamily accuracy | 0.447 | 0.453 |
| Subfamily macro-F1 | 0.213 | 0.232 |

On labels the model **ties or is slightly below** plain logistic regression.
The embedding is not a better classifier; its advantage is retrieval.

Full tables: `reports/eval_final_test.md`, `reports/eval_final_val.md`,
`reports/significance_final_test.md`.

### Ablations (validation split, 8 epochs, one seed each)

| Variant | Mean retrieval MRR change |
|---|---|
| without cross-database pairs in training | about -0.05 |
| without the chemistry prior | -0.005 |
| without note-word embeddings | -0.002 |
| 0 or 2 attention layers instead of 1 | -0.004 to -0.005 |

Only the cross-database pairs clearly matter. The chemistry prior, word
embeddings and layer count are within single-seed noise: **they are kept, but no
benefit from them has been demonstrated.** Full table: `reports/hone_ablations.md`.
These ablations were run before the name-parser fix described below.

### Protocol notes

- Checkpoints were selected on validation only, with the score
  0.6 x mean retrieval MRR + 0.4 x mean(accord mAP, subfamily accuracy).
- The test split was evaluated twice: once before and once after a data fix
  (836 truncated Parfumo names, which changed the matched pairs from 6,836 to
  6,931 and so changed the test set slightly). No modelling choice was made from
  test numbers. Earlier test results are kept in
  `reports/eval_final_test_before_namefix.md`; they differ by less than 0.015
  MRR on every task.
- `reports/before_hone_val.md` holds validation numbers from before the
  fabricated rows were removed. That validation set is different, so those
  numbers are not directly comparable with the current ones.

## Behaviour on known pairs

A small sanity check, not a benchmark:

| Pair | Cosine | Rank among 44,658 |
|---|---|---|
| Creed Aventus vs Armaf Club de Nuit Intense Man (well-known clone) | 0.830 | 26 |
| Baccarat Rouge 540 vs Ariana Grande Cloud (often called similar) | 0.152 | 5,727 |
| Creed Aventus vs Montblanc Explorer (often called similar) | 0.132 | 9,217 |
| Dior Sauvage EdT vs Bleu de Chanel (same genre) | 0.185 | 6,243 |
| Tom Ford Tobacco Vanille vs Chanel N°5 (unrelated) | -0.258 | 43,558 |

This shows the main limit directly: where a clone publishes the same notes as
the original, the model finds it; where two perfumes smell alike but publish
different notes, it does not.

## Limitations

- **Listed notes, not smell.** Note pyramids are marketing descriptions. They
  omit concentrations, most aroma chemicals and anything the brand chooses not
  to mention. Similarity here means "described similarly".
- **No human smell-similarity ground truth** was used or is available. The
  retrieval test measures robustness to how notes are written, which is a proxy.
- **Label quality.** Accords are community votes; family labels come from a
  second database of unverified origin.
- **Coverage.** The Parfumo snapshot is incomplete and ends in 2024. Notes
  listed by fewer than three catalog perfumes are not recognised by the text
  search parser; rare notes fall back to their words and chemistry profile.
- **Residual bad data.** About 2% of Parfumo rows may still be fabricated.
- **Single seed.** All reported numbers come from one training run.
- **Text search is a parser, not a language model.** It recognises notes,
  accords, negations ("no", "without", "not too") and a few filters; other words
  are reported back as unrecognised.
- **English only.**

## Licensing and use

The code is MIT-licensed (see `LICENSE`). That does not extend to the data or
settle the status of the trained weights.

The catalog is built from scraped, community-edited data with unclear reuse
terms, and two chemistry sources are for research use. Treat the packaged
catalog and model as suitable for personal and research use. Commercial use or
redistribution needs a licence review of each source, or retraining on data you
have rights to; the pipeline in the README works with any tables in the same
format.
