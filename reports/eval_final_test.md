### Held-out results (test split)

**parfumo_to_doevent** - 691 queries, gallery of 2,566 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.838 | 0.913 | 0.936 | 0.872 | 1 |
| jaccard | 0.809 | 0.891 | 0.904 | 0.845 | 1 |
| tfidf_notes | 0.748 | 0.871 | 0.909 | 0.807 | 1 |
| tfidf_words | 0.796 | 0.887 | 0.920 | 0.842 | 1 |
| lsa | 0.805 | 0.896 | 0.916 | 0.849 | 1 |

**doevent_to_parfumo** - 699 queries, gallery of 2,637 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.810 | 0.900 | 0.930 | 0.856 | 1 |
| jaccard | 0.738 | 0.856 | 0.894 | 0.795 | 1 |
| tfidf_notes | 0.561 | 0.745 | 0.807 | 0.646 | 1 |
| tfidf_words | 0.605 | 0.787 | 0.834 | 0.690 | 1 |
| lsa | 0.775 | 0.887 | 0.900 | 0.826 | 1 |

**parfumo_to_doevent_full_gallery** - 691 queries, gallery of 26,319 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.721 | 0.844 | 0.867 | 0.779 | 1 |
| jaccard | 0.670 | 0.823 | 0.849 | 0.739 | 1 |
| tfidf_notes | 0.609 | 0.763 | 0.802 | 0.681 | 1 |
| tfidf_words | 0.663 | 0.805 | 0.839 | 0.727 | 1 |
| lsa | 0.676 | 0.821 | 0.849 | 0.739 | 1 |

**doevent_to_parfumo_full_gallery** - 699 queries, gallery of 26,666 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.685 | 0.844 | 0.868 | 0.758 | 1 |
| jaccard | 0.589 | 0.760 | 0.793 | 0.668 | 1 |
| tfidf_notes | 0.413 | 0.575 | 0.639 | 0.495 | 2 |
| tfidf_words | 0.451 | 0.634 | 0.682 | 0.537 | 2 |
| lsa | 0.637 | 0.805 | 0.843 | 0.716 | 1 |

**Label prediction from notes**

| Method | Accord mAP | Accord P@1 | Accord nDCG@5 | Family acc | Subfamily acc | Subfamily macro-F1 |
|---|---|---|---|---|---|---|
| scentvec | 0.781 | 0.889 | 0.730 | 0.639 | 0.447 | 0.213 |
| logreg_bag_of_notes | 0.782 | 0.887 | 0.729 | 0.640 | 0.453 | 0.232 |
