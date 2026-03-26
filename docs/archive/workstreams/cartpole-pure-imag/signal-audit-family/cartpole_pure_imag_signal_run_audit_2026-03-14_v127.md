# controller Signal Comparison

- runs: `4`

## Comparison

| run | peak_eval | final_eval | eval_drop | post_peak_trust | post_peak_trigger_eff | post_peak_handoff_need | post_peak_freshness | likely_causes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp_seed42_v123_release_tail_sustain_only_2500_20260314_01 | 355.8 | 191.6 | -164.2 | 0.408 | 0.495 | 0.619 | 0.574 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |
| exp_seed42_v125_release_tail_sustain_handoffblock100_2500_20260314_01 | 355.8 | 162.2 | -193.6 | 0.418 | 0.528 | 0.600 | 0.584 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |
| exp_seed42_v126_release_tail_sustain_handoffblock100_confirm2_2500_20260314_01 | 355.8 | 162.2 | -193.6 | 0.418 | 0.528 | 0.600 | 0.584 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |
| exp_seed42_v127_landing_guard_on_v125line_2500_20260314_01 | 500.0 | 182.4 | -317.6 | 0.350 | 0.132 | 0.416 | 0.439 | post_peak_imagination_trust_low, post_peak_trigger_effectiveness_low, post_peak_model_freshness_low |

## Per Run
### exp_seed42_v123_release_tail_sustain_only_2500_20260314_01

# controller Signal Audit

- run_dir: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v123_release_tail_sustain_only_2500_20260314_01`
- metric_rows: `50`
- trace_rows: `50`
- eval: peak `355.8` @ step `750` -> final `191.6` @ step `2500` drop `-164.2`

## Signal Stats
- `imagination_trust` mean=`0.472` min=`0.1412912368774414`@`2300` max=`0.7933634584148725`@`600` last=`0.3325525159637134`
- `trigger_effectiveness` mean=`0.420` min=`0.06135734096169472`@`2300` max=`0.8326994881033898`@`2450` last=`0.7985176265239716`
- `handoff_necessity` mean=`0.492` min=`0.08509535623921288`@`150` max=`1.0`@`1950` last=`0.7521622177627352`
- `model_freshness` mean=`0.649` min=`0.31672250371509125`@`2300` max=`0.8957958539765741`@`900` last=`0.558227446998987`

## Post-Peak Signal Stats
- `imagination_trust` mean=`0.408` focus=`0.1412912368774414` @ step `2300` stage `persistence`
- `trigger_effectiveness` mean=`0.495` focus=`0.06135734096169472` @ step `2300` stage `persistence`
- `handoff_necessity` mean=`0.619` focus=`1.0` @ step `1950` stage `persistence`
- `model_freshness` mean=`0.574` focus=`0.31672250371509125` @ step `2300` stage `persistence`

## Stage Signal Stats
- `idle` rows=`15` trust=`0.638` trigger_eff=`0.226` handoff_need=`0.166` freshness=`0.841` release=`0.000` online_adv=`1.751`
- `persistence` rows=`4` trust=`0.210` trigger_eff=`0.088` handoff_need=`1.000` freshness=`0.397` release=`0.000` online_adv=`-6.634`
- `persistence_release` rows=`4` trust=`0.298` trigger_eff=`0.152` handoff_need=`0.850` freshness=`0.515` release=`0.000` online_adv=`-5.993`
- `post_entry` rows=`4` trust=`0.751` trigger_eff=`0.602` handoff_need=`0.161` freshness=`0.849` release=`0.000` online_adv=`-0.465`
- `post_entry_soft` rows=`5` trust=`0.534` trigger_eff=`0.371` handoff_need=`0.325` freshness=`0.613` release=`0.000` online_adv=`-3.434`
- `trigger` rows=`18` trust=`0.350` trigger_eff=`0.689` handoff_need=`0.690` freshness=`0.542` release=`0.872` online_adv=`-5.170`

## Eval Checkpoints
- step `250` eval=`15.9` stage=`idle` trust=`0.561` trigger_eff=`0.227` handoff_need=`0.108` freshness=`0.846`
- step `500` eval=`107.6` stage=`idle` trust=`0.579` trigger_eff=`0.227` handoff_need=`0.178` freshness=`0.824`
- step `750` eval=`355.8` stage=`idle` trust=`0.681` trigger_eff=`0.206` handoff_need=`0.211` freshness=`0.805`
- step `1000` eval=`183.5` stage=`post_entry_soft` trust=`0.729` trigger_eff=`0.542` handoff_need=`0.168` freshness=`0.811`
- step `1250` eval=`177.4` stage=`trigger` trust=`0.336` trigger_eff=`0.152` handoff_need=`0.543` freshness=`0.399`
- step `1500` eval=`167.1` stage=`trigger` trust=`0.624` trigger_eff=`0.774` handoff_need=`0.263` freshness=`0.742`
- step `1750` eval=`175.5` stage=`trigger` trust=`0.281` trigger_eff=`0.738` handoff_need=`0.787` freshness=`0.486`
- step `2000` eval=`125.6` stage=`trigger` trust=`0.251` trigger_eff=`0.756` handoff_need=`0.846` freshness=`0.492`
- step `2250` eval=`132.1` stage=`trigger` trust=`0.415` trigger_eff=`0.488` handoff_need=`0.655` freshness=`0.568`
- step `2500` eval=`191.6` stage=`trigger` trust=`0.333` trigger_eff=`0.799` handoff_need=`0.752` freshness=`0.558`

## Likely Causes
- `post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness`

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

### exp_seed42_v126_release_tail_sustain_handoffblock100_confirm2_2500_20260314_01

# controller Signal Audit

- run_dir: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v126_release_tail_sustain_handoffblock100_confirm2_2500_20260314_01`
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

### exp_seed42_v127_landing_guard_on_v125line_2500_20260314_01

# controller Signal Audit

- run_dir: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v127_landing_guard_on_v125line_2500_20260314_01`
- metric_rows: `50`
- trace_rows: `50`
- eval: peak `500.0` @ step `1500` -> final `182.4` @ step `2500` drop `-317.6`

## Signal Stats
- `imagination_trust` mean=`0.524` min=`0.11824923753738403`@`2000` max=`0.8390786518653233`@`950` last=`0.23575063198804855`
- `trigger_effectiveness` mean=`0.216` min=`0.0`@`1850` max=`0.5625754475593567`@`1400` last=`0.0`
- `handoff_necessity` mean=`0.282` min=`0.08509535623921288`@`150` max=`0.5276953763431974`@`2000` last=`0.4922421799765693`
- `model_freshness` mean=`0.647` min=`0.2950212510757976`@`2000` max=`0.9168836527417102`@`950` last=`0.3386819325296415`

## Post-Peak Signal Stats
- `imagination_trust` mean=`0.350` focus=`0.11824923753738403` @ step `2000` stage `trigger`
- `trigger_effectiveness` mean=`0.132` focus=`0.0` @ step `1850` stage `trigger`
- `handoff_necessity` mean=`0.416` focus=`0.5276953763431974` @ step `2000` stage `trigger`
- `model_freshness` mean=`0.439` focus=`0.2950212510757976` @ step `2000` stage `trigger`

## Stage Signal Stats
- `idle` rows=`21` trust=`0.661` trigger_eff=`0.217` handoff_need=`0.165` freshness=`0.835` release=`0.000` online_adv=`1.033`
- `trigger` rows=`29` trust=`0.424` trigger_eff=`0.214` handoff_need=`0.367` freshness=`0.511` release=`0.000` online_adv=`-5.832`

## Eval Checkpoints
- step `250` eval=`15.9` stage=`idle` trust=`0.561` trigger_eff=`0.227` handoff_need=`0.108` freshness=`0.846`
- step `500` eval=`107.6` stage=`idle` trust=`0.579` trigger_eff=`0.227` handoff_need=`0.178` freshness=`0.824`
- step `750` eval=`355.8` stage=`idle` trust=`0.681` trigger_eff=`0.206` handoff_need=`0.211` freshness=`0.805`
- step `1000` eval=`218.4` stage=`idle` trust=`0.738` trigger_eff=`0.198` handoff_need=`0.142` freshness=`0.835`
- step `1250` eval=`329.7` stage=`trigger` trust=`0.512` trigger_eff=`0.343` handoff_need=`0.298` freshness=`0.603`
- step `1500` eval=`500.0` stage=`trigger` trust=`0.699` trigger_eff=`0.475` handoff_need=`0.192` freshness=`0.759`
- step `1750` eval=`500.0` stage=`trigger` trust=`0.454` trigger_eff=`0.205` handoff_need=`0.364` freshness=`0.489`
- step `2000` eval=`23.1` stage=`trigger` trust=`0.118` trigger_eff=`0.000` handoff_need=`0.528` freshness=`0.295`
- step `2250` eval=`130.3` stage=`trigger` trust=`0.133` trigger_eff=`0.000` handoff_need=`0.510` freshness=`0.306`
- step `2500` eval=`182.4` stage=`trigger` trust=`0.236` trigger_eff=`0.000` handoff_need=`0.492` freshness=`0.339`

## Likely Causes
- `post_peak_imagination_trust_low, post_peak_trigger_effectiveness_low, post_peak_model_freshness_low`
