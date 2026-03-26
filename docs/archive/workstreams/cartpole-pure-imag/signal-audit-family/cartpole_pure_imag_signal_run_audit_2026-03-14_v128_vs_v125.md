# controller Signal Comparison

- runs: `2`

## Comparison

| run | peak_eval | final_eval | eval_drop | post_peak_trust | post_peak_trigger_eff | post_peak_handoff_need | post_peak_freshness | likely_causes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp_seed42_v125_release_tail_sustain_handoffblock100_2500_20260314_01 | 355.8 | 162.2 | -193.6 | 0.418 | 0.528 | 0.600 | 0.584 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |
| exp_seed42_v128_landing_guard_true_exact_v125line_2500_20260314_01 | 355.8 | 23.8 | -332.0 | 0.405 | 0.529 | 0.618 | 0.576 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |

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
