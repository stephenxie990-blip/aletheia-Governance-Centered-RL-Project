# CartPole Pure-Imag 失稳起点对照表

## `v130` vs `v145`

下面这张表只回答一个问题：

坏掉到底是从哪一刻开始的。

| 信号 | `v130` 首次出现 | `v145` 首次出现 | 解释 |
| --- | --- | --- | --- |
| `wm_cont - imag_cont > 0.20` | `1300 @ persistence_release` | `1350 @ trigger` | 坏 run 更早出现 imagined continue 脱锚 |
| `imag_cont < 0.75` | `1650 @ persistence` | `1350 @ trigger` | 坏 run 的 horizon 塌缩更早 |
| `value - return > 8` | `1700 @ persistence` | `1350 @ trigger` | 坏 run 的 critic 乐观滞后更早出现 |
| `online_adv < -8` | `1700 @ persistence` | `1350 @ trigger` | 坏 run 更早进入系统性负 online advantage |
| `adv > 0` | `1050 @ post_entry` | `1050 @ post_entry` | 两者早期都能有正向 corridor |
| 进入 `persistence` | `1650` | `1400` | 坏 run 更早掉入长期保护态 |

## 读表结论

`v145` 与 `v130` 的真正分界不是“进不进 persistence”，而是：

1. `v145` 在还没进入 `persistence` 之前，imagined continue、horizon、`value-return gap` 就已经先坏了；
2. 它进入 `persistence` 时，actor 的 `adv` 已经没有被修回正值；
3. 所以后面的保护更多是在维持一个坏状态，而不是帮助恢复。

`v130` 则相反：

1. 它也会出现 imagined continue 脱锚；
2. 但在进入严重坏区之前，`persistence_release` 先把 actor 正向 corridor 暂时保住了；
3. 因此它还能从坏区里短暂拉回来。
