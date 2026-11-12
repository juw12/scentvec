### Held-out results (val split)

**parfumo_to_doevent** - 665 queries, gallery of 2,619 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.860 | 0.914 | 0.931 | 0.887 | 1 |
| jaccard | 0.821 | 0.901 | 0.922 | 0.860 | 1 |
| tfidf_notes | 0.764 | 0.859 | 0.901 | 0.809 | 1 |
| tfidf_words | 0.794 | 0.881 | 0.911 | 0.835 | 1 |
| lsa | 0.820 | 0.893 | 0.910 | 0.854 | 1 |

**doevent_to_parfumo** - 669 queries, gallery of 2,718 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.809 | 0.895 | 0.918 | 0.851 | 1 |
| jaccard | 0.732 | 0.861 | 0.882 | 0.792 | 1 |
| tfidf_notes | 0.550 | 0.747 | 0.800 | 0.636 | 1 |
| tfidf_words | 0.590 | 0.782 | 0.837 | 0.673 | 1 |
| lsa | 0.779 | 0.880 | 0.898 | 0.825 | 1 |

**parfumo_to_doevent_full_gallery** - 665 queries, gallery of 26,319 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.777 | 0.860 | 0.889 | 0.816 | 1 |
| jaccard | 0.702 | 0.829 | 0.851 | 0.758 | 1 |
| tfidf_notes | 0.642 | 0.758 | 0.805 | 0.698 | 1 |
| tfidf_words | 0.678 | 0.794 | 0.835 | 0.731 | 1 |
| lsa | 0.707 | 0.824 | 0.853 | 0.762 | 1 |

**doevent_to_parfumo_full_gallery** - 669 queries, gallery of 26,666 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.706 | 0.846 | 0.867 | 0.769 | 1 |
| jaccard | 0.596 | 0.762 | 0.806 | 0.667 | 1 |
| tfidf_notes | 0.398 | 0.546 | 0.614 | 0.472 | 3 |
| tfidf_words | 0.426 | 0.589 | 0.643 | 0.504 | 2 |
| lsa | 0.661 | 0.807 | 0.848 | 0.726 | 1 |

**Label prediction from notes**

| Method | Accord mAP | Accord P@1 | Accord nDCG@5 | Family acc | Subfamily acc | Subfamily macro-F1 |
|---|---|---|---|---|---|---|
| scentvec | 0.781 | 0.888 | 0.726 | 0.635 | 0.444 | 0.210 |
| logreg_bag_of_notes | 0.780 | 0.882 | 0.724 | 0.641 | 0.438 | 0.226 |
