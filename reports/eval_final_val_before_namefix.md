### Held-out results (val split)

**parfumo_to_doevent** - 659 queries, gallery of 2,619 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.865 | 0.915 | 0.933 | 0.890 | 1 |
| jaccard | 0.819 | 0.900 | 0.921 | 0.859 | 1 |
| tfidf_notes | 0.762 | 0.857 | 0.900 | 0.807 | 1 |
| tfidf_words | 0.792 | 0.880 | 0.909 | 0.833 | 1 |
| lsa | 0.818 | 0.892 | 0.909 | 0.853 | 1 |

**doevent_to_parfumo** - 663 queries, gallery of 2,732 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.802 | 0.905 | 0.925 | 0.849 | 1 |
| jaccard | 0.733 | 0.860 | 0.881 | 0.792 | 1 |
| tfidf_notes | 0.546 | 0.744 | 0.795 | 0.632 | 1 |
| tfidf_words | 0.585 | 0.783 | 0.834 | 0.669 | 1 |
| lsa | 0.781 | 0.879 | 0.896 | 0.825 | 1 |

**parfumo_to_doevent_full_gallery** - 659 queries, gallery of 26,319 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.771 | 0.866 | 0.883 | 0.812 | 1 |
| jaccard | 0.703 | 0.827 | 0.850 | 0.758 | 1 |
| tfidf_notes | 0.640 | 0.756 | 0.801 | 0.696 | 1 |
| tfidf_words | 0.678 | 0.792 | 0.833 | 0.731 | 1 |
| lsa | 0.706 | 0.822 | 0.850 | 0.761 | 1 |

**doevent_to_parfumo_full_gallery** - 663 queries, gallery of 26,666 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.704 | 0.843 | 0.867 | 0.767 | 1 |
| jaccard | 0.597 | 0.762 | 0.804 | 0.666 | 1 |
| tfidf_notes | 0.397 | 0.543 | 0.612 | 0.471 | 3 |
| tfidf_words | 0.425 | 0.588 | 0.641 | 0.502 | 3 |
| lsa | 0.665 | 0.810 | 0.843 | 0.728 | 1 |

**Label prediction from notes**

| Method | Accord mAP | Accord P@1 | Accord nDCG@5 | Family acc | Subfamily acc | Subfamily macro-F1 |
|---|---|---|---|---|---|---|
| scentvec | 0.780 | 0.887 | 0.724 | 0.638 | 0.447 | 0.213 |
| logreg_bag_of_notes | 0.780 | 0.883 | 0.724 | 0.641 | 0.439 | 0.226 |
