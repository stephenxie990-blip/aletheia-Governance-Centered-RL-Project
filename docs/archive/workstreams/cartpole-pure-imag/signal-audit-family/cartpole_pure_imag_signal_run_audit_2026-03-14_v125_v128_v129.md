# controller Signal Comparison

- runs: `3`

## Comparison

| run | peak_eval | final_eval | eval_drop | post_peak_trust | post_peak_trigger_eff | post_peak_handoff_need | post_peak_freshness | likely_causes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp_seed42_v125_release_tail_sustain_handoffblock100_2500_20260314_01 | 355.8 | 162.2 | -193.6 | 0.418 | 0.528 | 0.600 | 0.584 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |
| exp_seed42_v128_landing_guard_true_exact_v125line_2500_20260314_01 | 355.8 | 23.8 | -332.0 | 0.405 | 0.529 | 0.618 | 0.576 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |
| exp_seed42_v129_landing_guard_geom_exact_v125line_2500_20260314_01 | 355.8 | 250.7 | -105.1 | 0.394 | 0.514 | 0.630 | 0.562 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |

## Per Run
### exp_seed42_v125_release_tail_sustain_handoffblock100_2500_20260314_01

# controller Signal Audit

- run_dir: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v125_release_tail_sustain_handoffblock100_2500_20260314_01`
- metric_rows: `50`
- trace_rows: `50`
- eval: peak `355.8` @ step `750` -> final `162.2` @ step `2500` drop `-193.6`

## Signal Stats
- `imagination_trust` mean=`0.479` min=`0.14683672189712524`@`2000` max=`0.7933634584148725`@`600` last=`0.23612314462661743`
- `trigger_effectiveness` mean=`0.444` min=`0.06564787849783897`@`2000` max=`0.8235658213496209`@`2400` last=`0.740600997209549`
- `handoff_necessity` mean=`0.478` min=`0.08509535623921288`@`150` max=`1.0`@`2000` last=`0.8582977281676398`
- `model_freshness` mean=`0.656` min=`0.32674005344841217`@`2000` max=`0.8957958539765741`@`900` last=`0.4776564867628945`

## Post-Peak Signal Stats
- `imagination_trust` mean=`0.418` focus=`0.14683672189712524` @ step `2000` stage `persistence`
- `trigger_effectiveness` mean=`0.528` focus=`0.06564787849783897` @ step `2000` stage `persistence`
- `handoff_necessity` mean=`0.600` focus=`1.0` @ step `2000` stage `persistence`
- `model_freshness` mean=`0.584` focus=`0.32674005344841217` @ step `2000` stage `persistence`

## Stage Signal Stats
- `idle` rows=`15` trust=`0.638` trigger_eff=`0.226` handoff_need=`0.166` freshness=`0.841` release=`0.000` online_adv=`1.751`
- `persistence` rows=`2` trust=`0.174` trigger_eff=`0.075` handoff_need=`1.000` freshness=`0.355` release=`0.000` online_adv=`-7.499`
- `persistence_release` rows=`4` trust=`0.298` trigger_eff=`0.152` handoff_need=`0.850` freshness=`0.515` release=`0.000` online_adv=`-5.993`
- `post_entry` rows=`4` trust=`0.751` trigger_eff=`0.602` handoff_need=`0.161` freshness=`0.849` release=`0.000` online_adv=`-0.465`
- `post_entry_soft` rows=`5` trust=`0.534` trigger_eff=`0.371` handoff_need=`0.325` freshness=`0.613` release=`0.000` online_adv=`-3.434`
- `trigger` rows=`20` trust=`0.357` trigger_eff=`0.689` handoff_need=`0.687` freshness=`0.549` release=`0.885` online_adv=`-5.002`

## Eval Checkpoints
- step `250` eval=`15.9` stage=`idle` trust=`0.561` trigger_eff=`0.227` handoff_need=`0.108` freshness=`0.846`
- step `500` eval=`107.6` stage=`idle` trust=`0.579` trigger_eff=`0.227` handoff_need=`0.178` freshness=`0.824`
- step `750` eval=`355.8` stage=`idle` trust=`0.681` trigger_eff=`0.206` handoff_need=`0.211` freshness=`0.805`
- step `1000` eval=`183.5` stage=`post_entry_soft` trust=`0.729` trigger_eff=`0.542` handoff_need=`0.168` freshness=`0.811`
- step `1250` eval=`177.4` stage=`trigger` trust=`0.336` trigger_eff=`0.152` handoff_need=`0.543` freshness=`0.399`
- step `1500` eval=`167.1` stage=`trigger` trust=`0.624` trigger_eff=`0.774` handoff_need=`0.263` freshness=`0.742`
- step `1750` eval=`175.5` stage=`trigger` trust=`0.281` trigger_eff=`0.738` handoff_need=`0.787` freshness=`0.486`
- step `2000` eval=`147.8` stage=`persistence` trust=`0.147` trigger_eff=`0.066` handoff_need=`1.000` freshness=`0.327`
- step `2250` eval=`145.7` stage=`trigger` trust=`0.282` trigger_eff=`0.789` handoff_need=`0.819` freshness=`0.530`
- step `2500` eval=`162.2` stage=`trigger` trust=`0.236` trigger_eff=`0.741` handoff_need=`0.858` freshness=`0.478`

## Likely Causes
- `post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness`

### exp_seed42_v128_landing_guard_true_exact_v125line_2500_20260314_01

# controller Signal Audit

- run_dir: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v128_landing_guard_true_exact_v125line_2500_20260314_01`
- metric_rows: `50`
- trace_rows: `50`
- eval: peak `355.8` @ step `750` -> final `23.8` @ step `2500` drop `-332.0`

## Signal Stats
- `imagination_trust` mean=`0.469` min=`0.1652384102344513`@`1900` max=`0.7933634584148725`@`600` last=`0.2596864938735962`
- `trigger_effectiveness` mean=`0.445` min=`0.07152900584042073`@`1900` max=`0.8174377024173737`@`1850` last=`0.7628149658441544`
- `handoff_necessity` mean=`0.491` min=`0.08509535623921288`@`150` max=`1.0`@`1900` last=`0.837852050198449`
- `model_freshness` mean=`0.651` min=`0.34515682904256717`@`1900` max=`0.8957958539765741`@`900` last=`0.5011161877082454`

## Post-Peak Signal Stats
- `imagination_trust` mean=`0.405` focus=`0.1652384102344513` @ step `1900` stage `persistence`
- `trigger_effectiveness` mean=`0.529` focus=`0.07152900584042073` @ step `1900` stage `persistence`
- `handoff_necessity` mean=`0.618` focus=`1.0` @ step `1900` stage `persistence`
- `model_freshness` mean=`0.576` focus=`0.34515682904256717` @ step `1900` stage `persistence`

## Stage Signal Stats
- `idle` rows=`15` trust=`0.638` trigger_eff=`0.226` handoff_need=`0.166` freshness=`0.841` release=`0.000` online_adv=`1.751`
- `persistence` rows=`2` trust=`0.218` trigger_eff=`0.093` handoff_need=`1.000` freshness=`0.412` release=`0.000` online_adv=`-6.329`
- `persistence_release` rows=`4` trust=`0.298` trigger_eff=`0.152` handoff_need=`0.850` freshness=`0.515` release=`0.000` online_adv=`-5.993`
- `post_entry` rows=`4` trust=`0.751` trigger_eff=`0.602` handoff_need=`0.161` freshness=`0.849` release=`0.000` online_adv=`-0.465`
- `post_entry_soft` rows=`5` trust=`0.534` trigger_eff=`0.371` handoff_need=`0.325` freshness=`0.613` release=`0.000` online_adv=`-3.434`
- `trigger` rows=`20` trust=`0.330` trigger_eff=`0.690` handoff_need=`0.718` freshness=`0.530` release=`0.885` online_adv=`-5.357`

## Eval Checkpoints
- step `250` eval=`15.9` stage=`idle` trust=`0.561` trigger_eff=`0.227` handoff_need=`0.108` freshness=`0.846`
- step `500` eval=`107.6` stage=`idle` trust=`0.579` trigger_eff=`0.227` handoff_need=`0.178` freshness=`0.824`
- step `750` eval=`355.8` stage=`idle` trust=`0.681` trigger_eff=`0.206` handoff_need=`0.211` freshness=`0.805`
- step `1000` eval=`183.5` stage=`post_entry_soft` trust=`0.729` trigger_eff=`0.542` handoff_need=`0.168` freshness=`0.811`
- step `1250` eval=`177.4` stage=`trigger` trust=`0.336` trigger_eff=`0.152` handoff_need=`0.543` freshness=`0.399`
- step `1500` eval=`167.1` stage=`trigger` trust=`0.624` trigger_eff=`0.774` handoff_need=`0.263` freshness=`0.742`
- step `1750` eval=`175.5` stage=`trigger` trust=`0.281` trigger_eff=`0.738` handoff_need=`0.787` freshness=`0.486`
- step `2000` eval=`70.6` stage=`trigger` trust=`0.279` trigger_eff=`0.774` handoff_need=`0.818` freshness=`0.521`
- step `2250` eval=`17.5` stage=`trigger` trust=`0.264` trigger_eff=`0.761` handoff_need=`0.832` freshness=`0.503`
- step `2500` eval=`23.8` stage=`trigger` trust=`0.260` trigger_eff=`0.763` handoff_need=`0.838` freshness=`0.501`

## Likely Causes
- `post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness`

### exp_seed42_v129_landing_guard_geom_exact_v125line_2500_20260314_01

# controller Signal Audit

- run_dir: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v129_landing_guard_geom_exact_v125line_2500_20260314_01`
- metric_rows: `50`
- trace_rows: `50`
- eval: peak `355.8` @ step `750` -> final `250.7` @ step `2500` drop `-105.1`

## Signal Stats
- `imagination_trust` mean=`0.461` min=`0.1405915379524231`@`2250` max=`0.7933634584148725`@`600` last=`0.1604757308959961`
- `trigger_effectiveness` mean=`0.434` min=`0.05671339444816112`@`2250` max=`0.8197525203227997`@`2450` last=`0.0665940798819065`
- `handoff_necessity` mean=`0.499` min=`0.08509535623921288`@`150` max=`1.0`@`1800` last=`1.0`
- `model_freshness` mean=`0.641` min=`0.3112690684331788`@`2250` max=`0.8957958539765741`@`900` last=`0.3355296471963326`

## Post-Peak Signal Stats
- `imagination_trust` mean=`0.394` focus=`0.1405915379524231` @ step `2250` stage `persistence`
- `trigger_effectiveness` mean=`0.514` focus=`0.05671339444816112` @ step `2250` stage `persistence`
- `handoff_necessity` mean=`0.630` focus=`1.0` @ step `1800` stage `persistence`
- `model_freshness` mean=`0.562` focus=`0.3112690684331788` @ step `2250` stage `persistence`

## Stage Signal Stats
- `idle` rows=`15` trust=`0.638` trigger_eff=`0.226` handoff_need=`0.166` freshness=`0.841` release=`0.000` online_adv=`1.751`
- `persistence` rows=`4` trust=`0.181` trigger_eff=`0.075` handoff_need=`1.000` freshness=`0.361` release=`0.000` online_adv=`-7.192`
- `persistence_release` rows=`4` trust=`0.298` trigger_eff=`0.152` handoff_need=`0.850` freshness=`0.515` release=`0.000` online_adv=`-5.993`
- `post_entry` rows=`4` trust=`0.751` trigger_eff=`0.602` handoff_need=`0.161` freshness=`0.849` release=`0.000` online_adv=`-0.465`
- `post_entry_soft` rows=`5` trust=`0.534` trigger_eff=`0.371` handoff_need=`0.325` freshness=`0.613` release=`0.000` online_adv=`-3.434`
- `trigger` rows=`18` trust=`0.328` trigger_eff=`0.729` handoff_need=`0.712` freshness=`0.526` release=`0.872` online_adv=`-5.527`

## Eval Checkpoints
- step `250` eval=`15.9` stage=`idle` trust=`0.561` trigger_eff=`0.227` handoff_need=`0.108` freshness=`0.846`
- step `500` eval=`107.6` stage=`idle` trust=`0.579` trigger_eff=`0.227` handoff_need=`0.178` freshness=`0.824`
- step `750` eval=`355.8` stage=`idle` trust=`0.681` trigger_eff=`0.206` handoff_need=`0.211` freshness=`0.805`
- step `1000` eval=`183.5` stage=`post_entry_soft` trust=`0.729` trigger_eff=`0.542` handoff_need=`0.168` freshness=`0.811`
- step `1250` eval=`177.4` stage=`trigger` trust=`0.336` trigger_eff=`0.152` handoff_need=`0.543` freshness=`0.399`
- step `1500` eval=`167.1` stage=`trigger` trust=`0.624` trigger_eff=`0.774` handoff_need=`0.263` freshness=`0.742`
- step `1750` eval=`64.9` stage=`trigger` trust=`0.264` trigger_eff=`0.758` handoff_need=`0.826` freshness=`0.494`
- step `2000` eval=`18.3` stage=`persistence` trust=`0.248` trigger_eff=`0.102` handoff_need=`1.000` freshness=`0.439`
- step `2250` eval=`25.1` stage=`persistence` trust=`0.141` trigger_eff=`0.057` handoff_need=`1.000` freshness=`0.311`
- step `2500` eval=`250.7` stage=`persistence` trust=`0.160` trigger_eff=`0.067` handoff_need=`1.000` freshness=`0.336`

## Likely Causes
- `post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness`
