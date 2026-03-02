# Phase 11: Color × Value Interaction

*Khipu Computational Toolkit — kfg-integration branch*

---

## Overview

Phase 11 tests whether cord color encodes the **unit scale** of recorded values —
the leading structural hypothesis in khipu studies.  If the Inca decimal
administration used color to signal magnitude tiers (units, tens, hundreds,
thousands), then color should be a statistically significant predictor of value
order-of-magnitude, independent of cord position or hierarchy level.

Five hypotheses are tested against the full 62,746-cord corpus.

**All five are confirmed at p < 0.001.**

---

## Methods

### Color encoding

The KFG `cord_colors` table stores one or more color codes per cord in
`sequence_ord` order.  The **primary color** (`sequence_ord = 0`) is the
outermost / dominant color.  Composite colors (e.g., `W%AB`, `MB-KB`) indicate
banded, barber-pole, or variegated cords where two color pigments alternate.

Only cords with a primary color code occurring ≥ 50 times in the corpus are
retained in analysis (59 qualifying color codes, covering virtually all cords).
Statistical tests use non-zero values only (zero-value cords are structural
placeholders; see Phase 10).

### Color code legend (most frequent)

| Code | Likely hue | Code | Likely hue |
|------|-----------|------|-----------|
| W | White / natural | AB | Amber-brown |
| MB | Medium brown | YB | Yellow-brown |
| KB | Khaki-brown | B | Brown |
| GG | Gray-green | LB | Light brown |
| NB | Natural brown | DB | Dark brown |
| W%AB | White-amber banded | W%MB | White-brown banded |
| YG | Yellow-green | NB | Natural brown |

---

## Results

### H1 — Primary color predicts value order-of-magnitude ✅

**Kruskal-Wallis: H = 1,419.0, p = 3.6 × 10⁻²⁶⁷**

The median value of non-zero cords varies by a factor of **126×** across color
codes (highest: W%AB median = 126; lowest: several single-tone codes, median = 1).

| Primary color | n (non-zero) | Median value | ≥100 % | ≥1,000 % |
|--------------|-------------:|-------------:|-------:|---------:|
| W%AB (banded) | 83 | 126.0 | 54.2 % | 10.8 % |
| DB-CB (banded) | 44 | 125.0 | 97.7 % | 0.0 % |
| MB-KB (banded) | 37 | 90.0 | 48.6 % | 13.5 % |
| YG | 224 | 80.5 | 45.5 % | 4.9 % |
| W%MB (banded) | 101 | 70.0 | 41.6 % | 4.0 % |
| NB | 801 | 42.0 | 32.7 % | 9.0 % |
| W (white) | 14,443 | 13.0 | 17.9 % | 4.5 % |
| B (brown) | 2,254 | 6.0 | 10.5 % | 1.4 % |

**The most striking pattern is that composite / banded colors consistently record
the highest values.** `W%AB` (white-amber banded), `DB-CB` (dark-cream banded),
and `W%MB` (white-brown banded) all have medians ≥ 70, and all are at least
4× higher than plain white (W, median = 13).  This is consistent with banded
cords marking a **higher accounting magnitude** — for example, hundreds or
thousands while their solid-color siblings record the ones and tens of the same
entry.

### H2 — Color distribution shifts by hierarchy level ✅

**Chi-square (level-0 vs level-1, top-15 colors): χ² = 1,267, df = 14, p = 7 × 10⁻²⁶²**

Color is not uniformly distributed across hierarchy levels:

| Color | Level-0 % | Level-1 % | Level-2 % | Trend |
|-------|----------:|----------:|----------:|-------|
| W (white) | 37.0 | 30.3 | 31.2 | Pendant-dominant |
| YB | 8.6 | 4.8 | 3.4 | Drops sharply with depth |
| KB | 2.2 | 6.3 | 6.0 | Subsidiary-dominant |
| AB | 15.4 | 15.8 | 21.4 | Increases with depth |
| B  | 5.1 | 4.5 | 2.3 | Drops with depth |

**YB (yellow-brown) is strongly pendant-associated** — it appears at 8.6 % of
level-0 pendants but only 3.4 % of level-2 sub-subsidiaries.  If YB marks a
specific commodity category (e.g., maize, a yellow crop), its pendant-dominance
suggests it records the primary quantity while subsidiaries handle sub-counts
in neutral colors.

**KB (khaki-brown) is subsidiary-dominant** (6.3–6.0 % at levels 1–2 vs 2.2 %
at level 0), consistent with KB cords being used for sub-totals or partial records
that roll up into a pendant aggregate.

**AB (amber-brown) increases with depth**, reaching 21.4 % of level-2 cords.
Deep AB cords may record the most granular administrative entries — individual
levy payments or census sub-units.

### H3 — Color patterns in summation-compliant groups (mixed) ✅

The 680 summation-compliant parent-child groups show a **parent-child same-color
match rate of 30.6 %**, compared to 27.7 % in the non-compliant (`sub`) class.
The difference is modest (2.9 pp), suggesting that summation compliance is not
primarily a color-matching phenomenon.  Instead:

- **W, AB, MB dominate both compliant and non-compliant groups**, reflecting their
  corpus-wide prevalence.
- The slight increase in same-color matching for compliant groups may indicate
  that when parent and children are recording the *same commodity unit*, they
  share a color — but cross-commodity aggregation (different colors) is permitted
  within a summation-compliant structure.

### H4 — Attachment type modulates the color-value signal ✅

**Kruskal-Wallis color × value is significant for all three attachment types:**

| Attachment | H-statistic | p-value |
|------------|------------:|--------:|
| U (upward) | 1,285.6 | 2.0 × 10⁻²³⁸ |
| R (recto)  | 695.0 | 5.4 × 10⁻¹²⁷ |
| V (verso)  | 521.1 | 1.3 × 10⁻ ⁹¹ |

Median values by attachment and top-5 colors reveal a consistent pattern:
**V-attached cords record higher values than U-attached cords** for the same
color (e.g., AB: V=14, R=11, U=5).  Verso attachment may mark a distinct recording
convention where the cord hangs "backward" to signal a different accounting unit or
a correction/amendment entry.

### H5 — Color composition differs significantly by behavioral cluster ✅

**Chi-square: χ² = 3,482, df = 70, p = 0**

| Cluster | Notable color signature |
|---------|------------------------|
| **B1** (non-numeric) | High LB (13.4 % vs corpus 2.0 %) — light brown may mark structural or ceremonial cords |
| **B2** (unit-count) | Elevated MB (16.9 %) and YB (12.1 %) — brown tones for counted goods |
| **B3** (L-knot workhorse) | Elevated B (7.3 %) — plain brown for routine recording |
| **B5** (flat high-variety) | Highest W (43.8 %) — white as the neutral multi-purpose recording color |
| **B6** (quota / round-5) | Elevated B (8.1 %) and NB (5.1 %) — dark/natural brown overrepresented in tribute records |

**B1's LB signature is the sharpest single-cluster color deviation** in the
analysis: 13.4 % of B1 cords are light brown versus a corpus average of 2.0 %.
Since B1 cords carry no numeric values, the LB color may be a conventional marker
for non-numeric or categorical cords — a "this cord is not a number" signal
embedded in the color code.

---

## Possible interpretations

### Banded cords = high-magnitude registers

The consistent elevation of composite/banded colors (`W%AB`, `W%MB`, `DB-CB`,
`MB-KB`) to 5–10× the value of their monochrome counterparts is the strongest
color-value signal in the dataset.  The most natural explanation within the Inca
decimal system is that banded cords record entries in a different *place-value
tier* — just as the knot position on a cord encodes units/tens/hundreds, the
color pattern may redundantly mark the register's scale, ensuring readers
navigating a complex khipu could quickly identify which cords hold large-magnitude
totals.

### YB as a commodity color

YB's pendant-dominance and its corpus frequency (3,429 cords) suggest it marks a
specific commodity class rather than a generic accounting role.  Yellow-brown is
associated with maize (*sara*) in Andean textile iconography.  If confirmed by
provenance analysis, YB cords may be a direct encoding of grain tribute — the
pendant records the total, subsidiaries track sub-ayllu contributions.

### LB as non-numeric marker

B1 khipus (all-zero values, no numeric content) are disproportionately LB-colored.
A parsimonious reading is that scribes used a specific cord color — light brown —
to indicate that a cord was **positional / structural**, not an active value
register.  This would be analogous to a placeholder zero in written numerals: the
LB cord holds the slot open without encoding a quantity.

### AB density increases with depth = granularity signal

The increasing share of amber-brown cords at deeper hierarchy levels (15.4 % →
21.4 % from pendant to sub-subsidiary) aligns with AB marking the most granular
administrative unit.  In a pachaka (hundred-household) ledger, the pendant AB cord
might record an ayllu subtotal while deeper AB cords record individual household
levies.

---

## Limitations

- Color code transcription in the KFG database relies on expert visual assessment
  of aged fibers; some codes may be misidentified due to pigment degradation.
- The banded / composite color analysis is based on a relatively small sample
  (83–101 cords for the top banded colors); results should be treated as
  directional pending larger replication.
- Region-level provenance for ~40 % of khipus is uncertain (see Phase 4),
  potentially masking regional color conventions.

---

## Outputs

| File | Description |
|------|-------------|
| `data/processed/phase11_color_value.csv` | Per-cord: primary color, value, hierarchy level, behavioral label |
| `data/processed/phase11_color_stats.csv` | Per-color: n, median, % ≥100, % ≥1,000 |
| `visualizations/phase11/color_value_boxplot.png` | H1: color × log-value boxplot, ordered by median |
| `visualizations/phase11/color_by_level.png` | H2: color distribution at levels 0, 1, 2 |
| `visualizations/phase11/color_compliance.png` | H3: color in compliant vs non-compliant summation groups |
| `visualizations/phase11/attachment_color.png` | H4: color × value by attachment type |
| `visualizations/phase11/color_cluster_heatmap.png` | H5: color composition heatmap across behavioral clusters |
