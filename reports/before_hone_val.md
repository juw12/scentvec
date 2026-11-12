### Held-out results (val split)

**parfumo_to_doevent** - 684 queries, gallery of 2,619 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.827 | 0.886 | 0.902 | 0.855 | 1 |
| jaccard | 0.772 | 0.854 | 0.879 | 0.812 | 1 |
| tfidf_notes | 0.690 | 0.803 | 0.846 | 0.742 | 1 |
| tfidf_words | 0.740 | 0.841 | 0.871 | 0.786 | 1 |
| lsa | 0.769 | 0.851 | 0.876 | 0.808 | 1 |

**doevent_to_parfumo** - 688 queries, gallery of 3,172 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.765 | 0.876 | 0.888 | 0.812 | 1 |
| jaccard | 0.692 | 0.815 | 0.836 | 0.750 | 1 |
| tfidf_notes | 0.509 | 0.688 | 0.741 | 0.588 | 1 |
| tfidf_words | 0.555 | 0.746 | 0.792 | 0.637 | 1 |
| lsa | 0.731 | 0.844 | 0.860 | 0.779 | 1 |

**parfumo_to_doevent_full_gallery** - 684 queries, gallery of 26,319 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.730 | 0.833 | 0.851 | 0.776 | 1 |
| jaccard | 0.655 | 0.782 | 0.804 | 0.711 | 1 |
| tfidf_notes | 0.563 | 0.683 | 0.730 | 0.622 | 1 |
| tfidf_words | 0.632 | 0.743 | 0.781 | 0.682 | 1 |
| lsa | 0.659 | 0.773 | 0.807 | 0.714 | 1 |

**doevent_to_parfumo_full_gallery** - 688 queries, gallery of 31,162 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.666 | 0.807 | 0.824 | 0.727 | 1 |
| jaccard | 0.547 | 0.711 | 0.760 | 0.618 | 1 |
| tfidf_notes | 0.353 | 0.496 | 0.567 | 0.425 | 6 |
| tfidf_words | 0.379 | 0.549 | 0.608 | 0.458 | 4 |
| lsa | 0.619 | 0.763 | 0.794 | 0.682 | 1 |

**Label prediction from notes**

| Method | Accord mAP | Accord P@1 | Accord nDCG@5 | Family acc | Subfamily acc | Subfamily macro-F1 |
|---|---|---|---|---|---|---|
| scentvec | 0.777 | 0.880 | 0.723 | 0.638 | 0.448 | 0.204 |
| logreg_bag_of_notes | 0.778 | 0.879 | 0.722 | 0.639 | 0.439 | 0.226 |
