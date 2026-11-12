### Paired bootstrap of the MRR difference (test split, 10,000 resamples)

Difference = scentvec minus baseline, same queries.  The interval covers query sampling only, not training-seed variation.

**parfumo_to_doevent** - 691 queries

| Baseline | scentvec MRR | Baseline MRR | Difference | 95% interval | Better / worse / equal |
|---|---|---|---|---|---|
| jaccard | 0.872 | 0.845 | +0.027 | +0.013 to +0.041 | 93 / 42 / 556 |
| tfidf_notes | 0.872 | 0.807 | +0.065 | +0.048 to +0.084 | 129 / 43 / 519 |
| tfidf_words | 0.872 | 0.842 | +0.030 | +0.015 to +0.046 | 97 / 42 / 552 |
| lsa | 0.872 | 0.849 | +0.023 | +0.009 to +0.036 | 89 / 42 / 560 |

**doevent_to_parfumo** - 699 queries

| Baseline | scentvec MRR | Baseline MRR | Difference | 95% interval | Better / worse / equal |
|---|---|---|---|---|---|
| jaccard | 0.856 | 0.795 | +0.061 | +0.042 to +0.079 | 133 / 35 / 531 |
| tfidf_notes | 0.856 | 0.646 | +0.210 | +0.182 to +0.238 | 259 / 36 / 404 |
| tfidf_words | 0.856 | 0.690 | +0.166 | +0.141 to +0.191 | 225 / 36 / 438 |
| lsa | 0.856 | 0.826 | +0.030 | +0.015 to +0.046 | 92 / 46 / 561 |

**parfumo_to_doevent_full_gallery** - 691 queries

| Baseline | scentvec MRR | Baseline MRR | Difference | 95% interval | Better / worse / equal |
|---|---|---|---|---|---|
| jaccard | 0.779 | 0.739 | +0.039 | +0.023 to +0.056 | 135 / 62 / 494 |
| tfidf_notes | 0.779 | 0.681 | +0.098 | +0.077 to +0.120 | 190 / 60 / 441 |
| tfidf_words | 0.779 | 0.727 | +0.051 | +0.034 to +0.070 | 148 / 64 / 479 |
| lsa | 0.779 | 0.739 | +0.039 | +0.024 to +0.055 | 137 / 63 / 491 |

**doevent_to_parfumo_full_gallery** - 699 queries

| Baseline | scentvec MRR | Baseline MRR | Difference | 95% interval | Better / worse / equal |
|---|---|---|---|---|---|
| jaccard | 0.758 | 0.668 | +0.090 | +0.069 to +0.112 | 202 / 61 / 436 |
| tfidf_notes | 0.758 | 0.495 | +0.263 | +0.233 to +0.293 | 331 / 50 / 318 |
| tfidf_words | 0.758 | 0.537 | +0.221 | +0.193 to +0.249 | 306 / 50 / 343 |
| lsa | 0.758 | 0.716 | +0.042 | +0.024 to +0.059 | 135 / 88 / 476 |
