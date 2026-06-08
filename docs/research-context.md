# Research Context — Lethe

Notes accumulated during substrate_zero analysis, relevant to Lethe design decisions.

---

## Physical analog: moiré devices (2026-06-06)

**Source:** arXiv:2510.21005 (Li, Rojas-Gatjens et al., Oct 2025) — "Photoinduced
Metal-to-Insulator Transitions in 2D Moiré Devices"

Ultrafast photoexcitation (energy-flux injection ≈ γ) drives a metal→correlated-insulator
transition in WS₂/WSe₂ moiré heterostructures. The induced insulating phase is metastable
with lifetime >1 µs. Key point: **the metastable memory effect only appears when the system
is near the correlated phase boundary** — it does not occur in the trivial metallic phase.

**Relevance to Phase 2 go/no-go gate:**

When `generalisation-experiment` runs on non-lattice substrates (FHN, Kuramoto, conductance),
a failure to observe λᵢ lift does NOT automatically mean the substrate is wrong. It may mean
the substrate was not swept into the equivalent of the correlated-phase boundary. The gate
verdict should distinguish:

- **λᵢ lift absent across all parameter regimes** → substrate-independent negative result,
  GO verdict is blocked
- **λᵢ lift absent only in trivial/inert regimes** → the substrate may be valid but the
  parameter sweep didn't find the boundary; retry near the boundary before calling NO

The FHN result (TE–TC robustly negative vs lattice-positive) shows that correlation geometry
is substrate-dependent even when co-activation is present. The moiré result extends this to
physical systems: memory is a boundary effect, not a bulk phase property.

**Citation for lethe paper/docs:** `\cite{li2025moire}` (already in substrate_zero refs.bib)

---

## Iontronic gel systems (2026-06-06)

**Source:** npj Soft Matter, DOI 10.1038/s44431-026-00027-8 — "Design principles of soft
integrated iontronics using gels"

Gel-based iontronic systems naturally exhibit hysteresis (memristive response), ionic
nonlinearity (concentration polarisation, Wien effect), and sustained electrochemical flux.
The reference list includes Chua's memristor, reservoir computing via ionic memristors,
fluidic memristors, and 2D nanofluidic long-term memory (Robin et al., Science 2023).

**Relevance:** If lethe's Phase 5 device survey includes soft/wet substrates, iontronic gels
are the most immediately testable physical substrate where all three conditions are present
by design. The "design principles" framing of this paper means there may already be parameter
mappings analogous to (α, β, γ).
