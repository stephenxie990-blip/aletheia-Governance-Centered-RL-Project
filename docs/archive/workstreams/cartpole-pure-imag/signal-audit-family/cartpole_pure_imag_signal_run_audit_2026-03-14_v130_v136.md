# controller Signal Comparison

- runs: `2`

## Comparison

| run | peak_eval | final_eval | eval_drop | post_peak_trust | post_peak_trigger_eff | post_peak_handoff_need | post_peak_freshness | likely_causes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp_seed42_v130_highwaterfloor_exact_v125line_2500_20260314_01 | 389.2 | 320.0 | -69.2 | 0.264 | 0.590 | 0.835 | 0.469 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |
| exp_seed42_v136_dropconfirmguard_exact_v125line_2500_20260314_01 | 355.8 | 56.9 | -298.9 | 0.292 | 0.411 | 0.701 | 0.483 | post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness |

## Per Run
### exp_seed42_v130_highwaterfloor_exact_v125line_2500_20260314_01

# controller Signal Audit

- run_dir: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v130_highwaterfloor_exact_v125line_2500_20260314_01`
- metric_rows: `50`
- trace_rows: `50`
- eval: peak `389.2` @ step `1750` -> final `320.0` @ step `2500` drop `-69.2`

## Signal Stats
- `imagination_trust` mean=`0.452` min=`0.13500531911849975`@`1700` max=`0.7933634584148725`@`600` last=`0.16998554468154908`
- `trigger_effectiveness` mean=`0.403` min=`0.05578359104692936`@`1700` max=`0.8156168967485428`@`2000` last=`0.06532489247620106`
- `handoff_necessity` mean=`0.511` min=`0.08509535623921288`@`150` max=`1.0`@`1700` last=`1.0`
- `model_freshness` mean=`0.629` min=`0.3072066459589534`@`1700` max=`0.8957958539765741`@`900` last=`0.3392801021883885`

## Post-Peak Signal Stats
- `imagination_trust` mean=`0.264` focus=`0.13667322397232057` @ step `1950` stage `persistence`
- `trigger_effectiveness` mean=`0.590` focus=`0.05950804464519024` @ step `1800` stage `persistence`
- `handoff_necessity` mean=`0.835` focus=`1.0` @ step `1800` stage `persistence`
- `model_freshness` mean=`0.469` focus=`0.3117176983687613` @ step `1950` stage `persistence`

## Stage Signal Stats
- `idle` rows=`15` trust=`0.638` trigger_eff=`0.226` handoff_need=`0.166` freshness=`0.841` release=`0.000` online_adv=`1.751`
- `persistence` rows=`6` trust=`0.160` trigger_eff=`0.066` handoff_need=`0.999` freshness=`0.334` release=`0.000` online_adv=`-7.694`
- `persistence_release` rows=`4` trust=`0.298` trigger_eff=`0.152` handoff_need=`0.850` freshness=`0.515` release=`0.000` online_adv=`-5.993`
- `post_entry` rows=`4` trust=`0.751` trigger_eff=`0.602` handoff_need=`0.161` freshness=`0.849` release=`0.000` online_adv=`-0.465`
- `post_entry_soft` rows=`5` trust=`0.534` trigger_eff=`0.371` handoff_need=`0.325` freshness=`0.613` release=`0.000` online_adv=`-3.434`
- `trigger` rows=`16` trust=`0.324` trigger_eff=`0.718` handoff_need=`0.711` freshness=`0.520` release=`0.857` online_adv=`-5.691`

## Eval Checkpoints
- step `250` eval=`15.9` stage=`idle` trust=`0.561` trigger_eff=`0.227` handoff_need=`0.108` freshness=`0.846`
- step `500` eval=`107.6` stage=`idle` trust=`0.579` trigger_eff=`0.227` handoff_need=`0.178` freshness=`0.824`
- step `750` eval=`355.8` stage=`idle` trust=`0.681` trigger_eff=`0.206` handoff_need=`0.211` freshness=`0.805`
- step `1000` eval=`183.5` stage=`post_entry_soft` trust=`0.729` trigger_eff=`0.542` handoff_need=`0.168` freshness=`0.811`
- step `1250` eval=`177.4` stage=`trigger` trust=`0.336` trigger_eff=`0.152` handoff_need=`0.543` freshness=`0.399`
- step `1500` eval=`167.1` stage=`trigger` trust=`0.624` trigger_eff=`0.774` handoff_need=`0.263` freshness=`0.742`
- step `1750` eval=`389.2` stage=`trigger` trust=`0.312` trigger_eff=`0.775` handoff_need=`0.760` freshness=`0.518`
- step `2000` eval=`117.9` stage=`trigger` trust=`0.464` trigger_eff=`0.816` handoff_need=`0.569` freshness=`0.612`
- step `2250` eval=`63.5` stage=`trigger` trust=`0.181` trigger_eff=`0.701` handoff_need=`0.912` freshness=`0.430`
- step `2500` eval=`320.0` stage=`persistence` trust=`0.170` trigger_eff=`0.065` handoff_need=`1.000` freshness=`0.339`

## Likely Causes
- `post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness`

### exp_seed42_v136_dropconfirmguard_exact_v125line_2500_20260314_01

# controller Signal Audit

- run_dir: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v136_dropconfirmguard_exact_v125line_2500_20260314_01`
- metric_rows: `50`
- trace_rows: `50`
- eval: peak `355.8` @ step `750` -> final `56.9` @ step `2500` drop `-298.9`

## Signal Stats
- `imagination_trust` mean=`0.388` min=`0.12151719331741334`@`1350` max=`0.7933634584148725`@`600` last=`0.15155104994773866`
- `trigger_effectiveness` mean=`0.360` min=`0.030994032695889476`@`1800` max=`0.7460784271359444`@`1450` last=`0.6795997843146325`
- `handoff_necessity` mean=`0.550` min=`0.08509535623921288`@`150` max=`0.9692012081676059`@`1350` last=`0.9387229945924547`
- `model_freshness` mean=`0.584` min=`0.2973977811402745`@`1350` max=`0.8957958539765741`@`900` last=`0.40507450001272893`

## Post-Peak Signal Stats
- `imagination_trust` mean=`0.292` focus=`0.12151719331741334` @ step `1350` stage `trigger`
- `trigger_effectiveness` mean=`0.411` focus=`0.030994032695889476` @ step `1800` stage `persistence`
- `handoff_necessity` mean=`0.701` focus=`0.9692012081676059` @ step `1350` stage `trigger`
- `model_freshness` mean=`0.483` focus=`0.2973977811402745` @ step `1350` stage `trigger`

## Stage Signal Stats
- `idle` rows=`15` trust=`0.638` trigger_eff=`0.226` handoff_need=`0.166` freshness=`0.841` release=`0.000` online_adv=`1.751`
- `persistence` rows=`5` trust=`0.219` trigger_eff=`0.047` handoff_need=`0.850` freshness=`0.399` release=`0.000` online_adv=`-6.681`
- `persistence_release` rows=`4` trust=`0.149` trigger_eff=`0.131` handoff_need=`0.850` freshness=`0.401` release=`0.000` online_adv=`-8.677`
- `post_entry` rows=`4` trust=`0.751` trigger_eff=`0.602` handoff_need=`0.161` freshness=`0.849` release=`0.000` online_adv=`-0.465`
- `post_entry_soft` rows=`4` trust=`0.566` trigger_eff=`0.400` handoff_need=`0.294` freshness=`0.648` release=`0.000` online_adv=`-3.080`
- `trigger` rows=`18` trust=`0.161` trigger_eff=`0.547` handoff_need=`0.864` freshness=`0.389` release=`0.829` online_adv=`-8.210`

## Eval Checkpoints
- step `250` eval=`15.9` stage=`idle` trust=`0.561` trigger_eff=`0.227` handoff_need=`0.108` freshness=`0.846`
- step `500` eval=`107.6` stage=`idle` trust=`0.579` trigger_eff=`0.227` handoff_need=`0.178` freshness=`0.824`
- step `750` eval=`355.8` stage=`idle` trust=`0.681` trigger_eff=`0.206` handoff_need=`0.211` freshness=`0.805`
- step `1000` eval=`183.5` stage=`post_entry_soft` trust=`0.729` trigger_eff=`0.542` handoff_need=`0.168` freshness=`0.811`
- step `1250` eval=`212.8` stage=`trigger` trust=`0.202` trigger_eff=`0.073` handoff_need=`0.727` freshness=`0.318`
- step `1500` eval=`75.2` stage=`trigger` trust=`0.159` trigger_eff=`0.680` handoff_need=`0.929` freshness=`0.411`
- step `1750` eval=`164.7` stage=`trigger` trust=`0.154` trigger_eff=`0.675` handoff_need=`0.934` freshness=`0.406`
- step `2000` eval=`198.1` stage=`persistence` trust=`0.263` trigger_eff=`0.066` handoff_need=`0.850` freshness=`0.453`
- step `2250` eval=`129.8` stage=`trigger` trust=`0.126` trigger_eff=`0.561` handoff_need=`0.626` freshness=`0.385`
- step `2500` eval=`56.9` stage=`trigger` trust=`0.152` trigger_eff=`0.680` handoff_need=`0.939` freshness=`0.405`

## Likely Causes
- `post_peak_imagination_trust_low, trigger_release_window_carries_high_handoff_pressure, persistence_release_inherits_low_trust, persistence_phase_low_model_freshness`
