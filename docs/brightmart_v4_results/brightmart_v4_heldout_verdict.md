# Brightmart v4 - verdict (brightmart-v4)

Preregistration: docs/brightmart_v4_prereg.md. nDCG@10, paired bootstrap 10,000, seed 7, Holm alpha 0.05.

Thresholds: `{"metric": "nDCG@10", "n_boot": 10000, "seed": 7, "alpha": 0.05, "c3_ratio": 0.5, "t4_s2_margin": -0.05, "t7_hhem_faithful": 0.5}`

## Main family (standard corpus, S3+S4 templates, vs R0)

- R2s-R0: delta +0.291, CI [+0.202, +0.380], p 0.0000, Holm sig, n 80
- R1h-R0: delta -0.019, CI [-0.081, +0.041], p 0.5502, Holm ns, n 80
- R3s-R0: delta +0.253, CI [+0.167, +0.338], p 0.0000, Holm sig, n 80
- R5p-R0: delta +0.400, CI [+0.309, +0.492], p 0.0000, Holm sig, n 80
- R6f-R0: delta +0.722, CI [+0.644, +0.794], p 0.0000, Holm sig, n 80
- R7f-R0: delta +0.283, CI [+0.219, +0.346], p 0.0000, Holm sig, n 80
- R7h-R0: delta +0.167, CI [+0.092, +0.237], p 0.0000, Holm sig, n 80

## T1 scale: PASS
- R2s - R0 (templates): delta +0.291, CI [+0.202, +0.380], p 0.0000, Holm sig, n 80
- R2s - R0 (paraphrases): delta +0.204, CI [+0.091, +0.316], p 0.0004, n 80

## T2 hierarchy without parent names: PASS
- R5p-R0 (noparent S4 templates): delta +0.735, CI [+0.640, +0.822], p 0.0000, Holm sig, n 40
- R2s-R0 (noparent S4 templates): delta +0.299, CI [+0.185, +0.416], p 0.0000, Holm sig, n 40

## T3 numeric/date conditions: PASS
- R6f-R0 (S3 numeric templates): delta +0.783, CI [+0.678, +0.877], p 0.0000, Holm sig, n 20
- R6f-R2s (S3 numeric templates): delta +0.765, CI [+0.654, +0.862], p 0.0000, Holm sig, n 20
- exploratory R6l-R0: delta +0.773, CI [+0.657, +0.874], p 0.0000, n 20
- exploratory R6l-R2s: delta +0.755, CI [+0.643, +0.854], p 0.0000, n 20

## T4 multi-hop fusion: PASS
- R7f: S2 delta -0.035, CI [-0.096, +0.027], p 0.2774, n 40; S3+S4 delta +0.283, CI [+0.219, +0.346], p 0.0000, Holm sig, n 80 -> FAIL
- R7h: S2 delta +0.108, CI [+0.058, +0.159], p 0.0000, n 40; S3+S4 delta +0.167, CI [+0.092, +0.237], p 0.0000, Holm sig, n 80 -> PASS

## T5 contextual chunks and link expansion
- R1h: delta -0.019, CI [-0.081, +0.041], p 0.5502, Holm ns, n 80 -> not kept
- R3s: delta +0.253, CI [+0.167, +0.338], p 0.0000, Holm sig, n 80 -> kept

## T6 authored minus extracted (R3s - R3x)
- S34_t: delta -0.001, CI [-0.018, +0.017], p 0.9430, n 80
- S1to4_t: delta +0.001, CI [-0.008, +0.011], p 0.7602, n 160
- authored_pairs: 1383
- extracted_pairs: 2792
- shared_pairs: 836
- precision: 0.2994269340974212
- recall: 0.6044830079537238
- extraction: {'role': 'graph_extraction', 'model_id': 'Qwen/Qwen3-4B-Instruct-2507', 'revision': 'cdbee75f17c01a7cc42f958dc650907174af0554', 'dtype': 'torch.float16', 'cuda': True, 'override': False, 'docs': 348, 'edges': 3101, 'parse_ok': 348, 'unknown_titles': 16, 'prompt_tokens': 948702, 'completion_tokens': 19765, 'seconds': 1983.9, 'batch_size': 4, 'corpus': 'results/corpus_v4'}

## S1 direct facts (templates, vs R0)
- R2s: delta -0.199, CI [-0.257, -0.140], p 0.0000, n 40
- R1h: delta +0.003, CI [-0.046, +0.050], p 0.9492, n 40
- R3s: delta -0.177, CI [-0.232, -0.122], p 0.0000, n 40
- R5p: delta -0.199, CI [-0.257, -0.140], p 0.0000, n 40
- R6f: delta -0.199, CI [-0.257, -0.140], p 0.0000, n 40
- R7f: delta -0.028, CI [-0.062, +0.000], p 0.0706, n 40
- R7h: delta -0.010, CI [-0.075, +0.049], p 0.7410, n 40
