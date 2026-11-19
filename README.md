# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mitsukikun/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mitsukikun/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every Grand Theft Auto VI trophy on PlayStation 5, its earn rate on Exophase at 2026-11-26 00:00 UTC — as five quantiles a trophy (5%, 25%, 50%, 75%, 95%) — from the house's recording of the trophy page up to the case's checkpoint. The program reads one case file (`/input/<id>.in`) and writes one forecast (`/output/<id>.out`), offline, in pure-stdlib Python 3.13, in milliseconds.

How it works: Exophase's page is a snapshot of two moving quantities — the players tracked (N) and the players holding each trophy (E_i) — and a rate is E_i / N. The forecaster groups the history into epochs (one per refresh), fits N's growth as a power law over the recent epochs (growth capped at 8x over the future window), and decomposes each refresh jump per trophy into `dE = p·dN + g·N_old·dt` — p the share of newly tracked players already holding the trophy, g the per-player hourly earn rate of the players already tracked. That separates dilution (new players arriving with few trophies) from progression (old players earning more over time), which a rate-only trend fit cannot do. The median rate is `100·E(target)/N(target)`; the spread combines a random walk, the fit's residual scatter, and N uncertainty. When the history shows no refresh at all (a page frozen since launch), it falls back to a stale-page dilution prior: the next refresh is expected to multiply the tracked players and dilute the rates toward what fresh joiners hold.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dr78ckwq&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
