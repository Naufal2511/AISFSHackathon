# Fine-tuned behavior classifier beh-dinov2s-ft-v1

`vit_small_patch14_dinov2.lvd142m`, last 4 blocks trained, best epoch 7/20 (picked on val accuracy), temperature 1.15, 10.9 ms/crop on mps.
Trained on 181 of our tasks + 9269 CBVD-5 crops; evaluated on 64 of our val tasks (split by source clip).

| model (our val split) | macro-F1 | accuracy | ECE |
|---|---|---|---|
| frozen probe (v1) | 0.353 | 0.641 | 0.151 |
| fine-tuned, raw | 0.370 | 0.688 | 0.129 |
| fine-tuned, temperature-scaled | 0.370 | 0.688 | 0.155 |

Chosen for deployment: **finetuned**

## Per class (val)

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| lying | 27 | 0.9 | 1.0 | 0.947 |
| standing | 14 | 0.5 | 0.786 | 0.611 |
| walking | 6 | 0.25 | 0.167 | 0.2 |
| feeding | 3 | 1.0 | 1.0 | 1.0 |
| drinking | 5 | 1.0 | 0.4 | 0.571 |
| ruminating (lying) | 1 | 0.0 | 0.0 | 0.0 |
| ruminating (standing) | 4 | 0.0 | 0.0 | 0.0 |
| social interaction | 3 | 0.0 | 0.0 | 0.0 |
| other / not visible | 1 | 0.0 | 0.0 | 0.0 |

## Confusion (rows truth, cols predicted)

```
 27   0   0   0   0   0   0   0   0   0
  0  11   2   0   0   0   0   0   1   0
  0   4   1   0   0   0   0   0   1   0
  0   0   0   3   0   0   0   0   0   0
  0   2   0   0   2   0   0   0   1   0
  1   0   0   0   0   0   0   0   0   0
  1   3   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0
  0   2   1   0   0   0   0   0   0   0
  1   0   0   0   0   0   0   0   0   0
```

## By camera

| group | n | macro-F1 | accuracy |
|---|---|---|---|
| cam1 | 52 | 0.379 | 0.731 |
| cam2 | 8 | 0.281 | 0.375 |
| cam3 | 4 | 0.600 | 0.750 |

## By light

| group | n | macro-F1 | accuracy |
|---|---|---|---|
| day | 56 | 0.425 | 0.696 |
| night/IR | 8 | 0.227 | 0.625 |

Note: the val split is small, so differences of a few points are noise.