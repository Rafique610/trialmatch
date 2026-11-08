# baseline_test vs finetuned_test

Rows: 206 (same rows, same decoding).

| Part | Metric | Before | After |
| --- | --- | --- | --- |
| overall | accuracy | 0.039 | 0.709 |
| overall | macro-F1 | 0.077 | 0.566 |
| overall | parse errors | 0 | 0 |
| overall | evidence micro-F1 | 0.157 | 0.579 |
| overall | evidence macro-F1 | 0.061 | 0.460 |
| inclusion | accuracy | 0.107 | 0.733 |
| inclusion | macro-F1 | 0.099 | 0.541 |
| inclusion | parse errors | 0 | 0 |
| inclusion | evidence micro-F1 | 0.217 | 0.626 |
| inclusion | evidence macro-F1 | 0.065 | 0.559 |
| exclusion | accuracy | 0.000 | 0.695 |
| exclusion | macro-F1 | 0.000 | 0.487 |
| exclusion | parse errors | 0 | 0 |
| exclusion | evidence micro-F1 | 0.137 | 0.478 |
| exclusion | evidence macro-F1 | 0.059 | 0.264 |

## Per-class F1 · inclusion

| Label | Support | Before | After |
| --- | --- | --- | --- |
| included | 27 | 0.312 | 0.792 |
| not included | 7 | 0.000 | 0.182 |
| not enough information | 38 | 0.000 | 0.791 |
| not applicable | 3 | 0.082 | 0.400 |

## Per-class F1 · exclusion

| Label | Support | Before | After |
| --- | --- | --- | --- |
| excluded | 0 | – | – |
| not excluded | 91 | 0.000 | 0.818 |
| not enough information | 28 | 0.000 | 0.200 |
| not applicable | 12 | 0.000 | 0.444 |
