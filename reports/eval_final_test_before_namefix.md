### Held-out results (test split)

**parfumo_to_doevent** - 683 queries, gallery of 2,566 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.845 | 0.918 | 0.931 | 0.878 | 1 |
| jaccard | 0.807 | 0.890 | 0.903 | 0.843 | 1 |
| tfidf_notes | 0.745 | 0.870 | 0.908 | 0.804 | 1 |
| tfidf_words | 0.795 | 0.886 | 0.919 | 0.841 | 1 |
| lsa | 0.804 | 0.893 | 0.915 | 0.848 | 1 |

**doevent_to_parfumo** - 691 queries, gallery of 2,661 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.812 | 0.907 | 0.926 | 0.857 | 1 |
| jaccard | 0.738 | 0.855 | 0.893 | 0.794 | 1 |
| tfidf_notes | 0.559 | 0.742 | 0.800 | 0.643 | 1 |
| tfidf_words | 0.601 | 0.784 | 0.831 | 0.687 | 1 |
| lsa | 0.771 | 0.886 | 0.899 | 0.823 | 1 |

**parfumo_to_doevent_full_gallery** - 683 queries, gallery of 26,319 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.735 | 0.859 | 0.884 | 0.792 | 1 |
| jaccard | 0.671 | 0.821 | 0.848 | 0.738 | 1 |
| tfidf_notes | 0.609 | 0.760 | 0.799 | 0.680 | 1 |
| tfidf_words | 0.665 | 0.802 | 0.836 | 0.728 | 1 |
| lsa | 0.676 | 0.821 | 0.848 | 0.739 | 1 |

**doevent_to_parfumo_full_gallery** - 691 queries, gallery of 26,666 records

| Method | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|
| scentvec | 0.689 | 0.847 | 0.873 | 0.763 | 1 |
| jaccard | 0.588 | 0.758 | 0.792 | 0.666 | 1 |
| tfidf_notes | 0.412 | 0.572 | 0.637 | 0.493 | 2 |
| tfidf_words | 0.450 | 0.632 | 0.683 | 0.535 | 2 |
| lsa | 0.634 | 0.806 | 0.844 | 0.714 | 1 |

**Label prediction from notes**

| Method | Accord mAP | Accord P@1 | Accord nDCG@5 | Family acc | Subfamily acc | Subfamily macro-F1 |
|---|---|---|---|---|---|---|
| scentvec | 0.780 | 0.892 | 0.730 | 0.638 | 0.459 | 0.217 |
| logreg_bag_of_notes | 0.782 | 0.887 | 0.730 | 0.639 | 0.453 | 0.231 |
