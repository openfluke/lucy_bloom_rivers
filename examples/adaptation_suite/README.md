# Adaptation suite — mid-stream task flip

Lucy menu **[17]**. Reproduces the dense mid-stream adaptation idea from menu **[2]**, then spreads it across every layer type × dtype × paradigm × SIMD path.

Question under test: when the target flips mid-run (`A → B → A'`), which update strategy keeps adapting while decisions keep flowing?

## Layout (3 Go files)

| File | Role |
|------|------|
| `config.go` | Update modes, paradigms (`QAT` / `Nat`), SIMD path matrix, `Config`, and `Scenario` |
| `run.go` | Single-path runner: phase flips, forward/train loop, window metrics, `Result` |
| `report.go` | Comparison tables — per-path scores, dtype winners, QAT/Nat × SIMD means, mode ranking, session manifest |

Menu wiring lives in `examples/seven_layer/adaptation_menu.go` (Lucy entry via `examples/adaptation_menu.go`).

### `config.go`

Defines the six update modes:

| Mode | Style | When it updates |
|------|--------|-----------------|
| `NormalBP` | `poly.Train` batch BP | Periodic (train interval) |
| `Step+BP` | Per-step backward | After every output |
| `Tween` | Target prop | Periodic batch |
| `TweenChain` | Chain-rule tween | Periodic batch |
| `StepTween` | Per-step tween | After every output |
| `StepTweenChain` | Per-step chain-rule tween | After every output |

Each path is `Paradigm × SIMD × Mode`. `AllPaths(nativeExact)` builds the full matrix (Native omitted when the dtype cannot run exact storage).

Default `Config`: 450 steps, 9 windows, LR `0.02`, train interval `50ms`, max batch `20`, recovery threshold `50%`, decision deadline `10ms`.

### `run.go`

`Run(wire, path, scenario, cfg)` deserializes one network, sets tiling / exact dtype / SIMD forward, then steps through three equal phases:

1. Phase **A** — first target
2. Phase **B** — flipped target
3. Phase **A'** — original target again

Scores each step against the phase target, applies the path’s update mode, tracks latency vs the 10ms deadline, then fills adaptation / throughput / availability / score metrics. `RegressionScenario` builds sin→cos→sin phase targets for any layer stack.

### `report.go`

Print helpers used by the menu after each layer finishes:

- path table (`Score`, thr/s, avail%, avg acc, Ph1/Ph2/Ph3)
- best path per dtype
- paradigm × SIMD mean scores
- update-mode ranking (mean score across paths)
- session manifest (PASS/FAIL per layer)

`Score = Throughput × Availability% × AvgAccuracy / 10,000`.

## What the suite sweeps

**Layers:** Dense, SwiGLU, MHA, CNN1/2/3, RNN, LSTM, Embedding, Residual

**Axes per layer:**

- dtype grid from the 7-layer / 56-layer builders
- paradigm: QAT (tiled) vs Native (exact), when supported
- forward: scalar vs SIMD (AVX2 / NEON)
- update mode: the six modes above

## Running

```bash
cd lucy_bloom_rivers
go run .
# choose [17]
```

Prompts:

1. Steps — `[1]` 150 smoke · `[2]` 450 default · `[3]` 900
2. Grid — `[1]` 1³ (7-layer) · `[2]` 2³ (56-layer)
3. Layer — `[0]` all · `[1]` Dense · …

Log tee:

- `lucy_testing_output/adaptation_suite.txt`

## Reference dense run (sibling benchmark [2])

The suite extends the dense Chase/Avoid/Chase benchmark. Latest captured dense run (same six modes × SIMD on/off):

- `lucy_testing_output/dense_adaptation_20260715_091343.txt`

Setup:

```text
Timeline: [Chase 5s] → [AVOID 5s] → [Chase 5s]
Network:  6-layer Dense (8→32→64→64→64→32→4)
Engine:   loom/poly VolumetricNetwork
SIMD:     forward only (training/backward stay scalar)
```

### Avg accuracy

| Mode | SIMD off | SIMD on |
|------|--------:|--------:|
| `NormalBP` | 29.4 | 29.5 |
| `Step+BP` | 42.7 | **48.1** |
| `Tween` | 14.7 | 18.7 |
| `TweenChain` | 21.6 | 28.0 |
| `StepTween` | 5.5 | 5.8 |
| `StepTweenChain` | 42.6 | 38.9 |

### Phase competence (Chase1 / Avoid / Chase2)

| Mode | SIMD | Chase1 | Avoid | Chase2 |
|------|------|-------:|------:|-------:|
| `NormalBP` | off | 0.0% | 88.1% | 0.0% |
| `NormalBP` | on | 0.0% | 88.4% | 0.0% |
| `Step+BP` | off | 2.0% | 85.6% | **40.6%** |
| `Step+BP` | on | 8.6% | **98.8%** | 37.0% |
| `Tween` | off | 4.2% | 39.8% | 0.0% |
| `Tween` | on | 0.3% | 55.9% | 0.0% |
| `TweenChain` | off | 0.0% | 64.9% | 0.0% |
| `TweenChain` | on | 0.0% | 84.0% | 0.0% |
| `StepTween` | off | 0.2% | 16.4% | 0.0% |
| `StepTween` | on | 0.1% | 17.3% | 0.0% |
| `StepTweenChain` | off | 1.8% | 97.7% | 28.4% |
| `StepTweenChain` | on | 1.7% | 93.7% | 21.5% |

### Operational snapshot

| Mode | SIMD | Thr/s | Avail% | Peak lat | Avg lat | Score |
|------|------|------:|-------:|---------:|--------:|------:|
| `NormalBP` | off | 9,630 | 77.0% | 25ms | 104µs | 2,178 |
| `NormalBP` | on | 9,692 | 77.0% | 29ms | 103µs | 2,197 |
| `Step+BP` | off | 2,273 | 100% | 8ms | 440µs | 971 |
| `Step+BP` | on | 3,403 | 100% | 8ms | 294µs | 1,638 |
| `Tween` | off | 15,284 | 89.2% | 11ms | 65µs | 1,999 |
| `Tween` | on | **40,944** | 90.2% | 15ms | 24µs | 6,919 |
| `TweenChain` | off | 13,753 | 79.9% | 35ms | 73µs | 2,378 |
| `TweenChain` | on | 37,752 | 83.9% | 18ms | 26µs | **8,862** |
| `StepTween` | off | 3,533 | 100% | 7ms | 283µs | 194 |
| `StepTween` | on | 4,015 | 100% | 6ms | 249µs | 234 |
| `StepTweenChain` | off | 1,646 | 100% | 8ms | 607µs | 701 |
| `StepTweenChain` | on | 2,105 | 100% | 10ms | 475µs | 819 |

### Takeaways from this dense run

- **Best live adaptation quality:** `Step+BP` (SIMD on) — highest avg accuracy, strong Avoid phase, only modes that meaningfully recover Chase2.
- **Best combined score:** `TweenChain` (SIMD on) — high throughput + decent Avoid competence.
- **Fastest raw stream:** `Tween` (SIMD on) — ~41k actions/s.
- **Step modes** (`Step+BP`, `StepTween*`) hold **100% availability** (no batch-train pauses); periodic modes trade some availability for volume.
- First task flip (Chase→Avoid) is where most modes recover; the second flip (Avoid→Chase) is still hard — only `Step+BP` / `StepTweenChain` keep non-zero Chase2 accuracy.

Dense [2] is the readable single-network story; [17] asks the same question for every layer family and dtype cell.
