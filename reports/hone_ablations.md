### Refinement-pass ablations (validation split, clean data, 8 epochs, seed 7)

| Run | Change | Epochs | Best epoch | Selection score | MRR parfumo→doevent | MRR doevent→parfumo | Accord mAP | Family acc | Subfamily acc |
|---|---|---|---|---|---|---|---|---|---|
| h_full | all inputs, 1 attention layer | 8 | 8 | 0.7631 | 0.8835 | 0.8474 | 0.7749 | 0.6369 | 0.4441 |
| h_nochem | without the odor-chemistry prior | 8 | 8 | 0.7595 (-0.0036) | 0.8770 (-0.0065) | 0.8433 (-0.0041) | 0.7701 (-0.0048) | 0.6403 (+0.0034) | 0.4471 (+0.0030) |
| h_notok | without note-word embeddings | 8 | 8 | 0.7629 (-0.0002) | 0.8838 (+0.0003) | 0.8429 (-0.0045) | 0.7742 (-0.0007) | 0.6342 (-0.0027) | 0.4502 (+0.0061) |
| h_nopairs | without cross-database pairs in training | 8 | 7 | 0.7312 (-0.0319) | 0.8380 (-0.0455) | 0.7849 (-0.0625) | 0.7746 (-0.0003) | 0.6369 (+0.0000) | 0.4467 (+0.0026) |
| h_l0 | 0 attention layers (pooled bag of notes) | 8 | 8 | 0.7596 (-0.0035) | 0.8781 (-0.0054) | 0.8432 (-0.0042) | 0.7755 (+0.0006) | 0.6369 (+0.0000) | 0.4406 (-0.0035) |
| h_l2 | 2 attention layers | 8 | 8 | 0.7609 (-0.0022) | 0.8760 (-0.0075) | 0.8470 (-0.0004) | 0.7766 (+0.0017) | 0.6376 (+0.0007) | 0.4437 (-0.0004) |

Differences in brackets are against `h_full`.  One seed per run: differences of a few
thousandths are within run-to-run noise and should not be read as effects.
