# LLM Usage Disclosure

## Tool Information

- **Tool**: Claude (claude.ai)
- **Provider**: Anthropic
- **Model**: Claude Sonnet 4.6
- **Dates of use**: April 28 – May 2, 2026


---

## Summary of Usage

Claude was used for the following purposes:
- Evaluating the scope and complexity of the project against the proposal requirements
- Reviewing the correctness and effectiveness of optimizer implementations
- Writing and debugging automated experiment and plotting scripts
- Providing feedback on experiment design and result interpretation

All outputs were reviewed, tested, and verified by the team before inclusion in the codebase or report. Code was run locally and outputs were checked against expected theoretical behavior. Written content was edited and rewritten by the team to reflect their own understanding.

---

## Prompt Log

### Prompt 1 — Project Scope and Complexity Review

**Date**: April 12, 2026

**Prompt**:
> "We have just initialized our GitHub repository for our ECE 509 term project on variance reduction methods for SGD. Our proposal outlines that we need to implement SVRG, SAGA, and SARAH and compare them against a vanilla SGD baseline. Given where we are right now, could you give us a detailed action plan for the project, including what order to implement the algorithms, what experiments to run, and what datasets we should use to properly evaluate our methods?"

**Output relied upon**:
Claude provided a week-by-week action plan breaking down implementation order, experiment design (convergence plots, step size sweep, batch size sweep), and dataset recommendations (libsvm a9a, mushrooms). It recommended using gradient evaluations as the fair x-axis metric and flagged that the comparison plot needed to use epoch-averaged loss rather than per-iteration loss to be interpretable.

**How it was used**:
The action plan was used as a guide for structuring the project timeline. The suggestion to use epoch-averaged loss was adopted. The dataset recommendation (a9a) was accepted and implemented. The gradient evaluation metric suggestion was noted but not implemented due to time constraints.

---

### Prompt 2 — Reviewing Code Effectiveness

**Date**: April 29, 2026

**Prompt**:
> "What do you think of the quality of this code with respect to the project proposal?" [attached Noah's implementation of SGD and SARAH]

**Output relied upon**:
Claude identified that the code was missing fair x-axis comparison (gradient evaluations vs epochs), that SAGA and SVRG were absent, that real datasets were not yet used, and that per-iteration loss logging produced noisy plots. It provided a structured gap analysis against the proposal's success criteria.

**How it was used**:
The gap analysis informed which components needed to be built next. The switch from per-iteration to epoch-averaged loss logging was implemented based on this feedback. The team used the identified gaps as a task list for the remaining implementation work.

---

### Prompt 3 — Automating the Learning Rate Sweep

**Date**: April 30, 2026

**Prompt**:
> "Help me do the step size sweep plotting script."

**Output relied upon**:
Claude wrote `sweep_lr.py`, a script that calls `main_v4.py` as a subprocess for each (method, learning rate) combination, caches results to avoid re-running completed experiments, and produces three plot types: a grid plot, an overlay plot (all methods per learning rate), and a per-method plot (all learning rates per method). The caching logic and subprocess structure were the key contributions.

**How it was used**:
The script was copied into `experiments/sweep_lr.py` and run directly. The overlay plot output was used in the report. The team verified the plots matched expected behavior from manual runs before including them.

---

### Prompt 4 — Automating the Batch Size Sweep

**Date**: April 30, 2026

**Prompt**:
> "Now do the batch size sweep script."

**Output relied upon**:
Claude wrote `sweep_bs.py` following the same structure as `sweep_lr.py`, with the addition of an automatic skip for SVRG at batch size 1 (flagged as too slow for large datasets). The script produces grid, overlay, and per-method plots saved to `plots/sweep_bs/`.

**How it was used**:
The script was copied into `experiments/sweep_bs.py` and run on the a9a dataset. The overlay plot was included in the report. The SVRG skip logic was verified manually to confirm correctness.

---

### Prompt 5 — Gradient Norm Logging and Visualization

**Date**: May 1, 2026

**Prompt**:
> "I was thinking if we should show how the gradient norm on every iteration differs compared to vanilla SGD — would this be useful? Please add gradient norm logging and write the plotting script for it."

**Output relied upon**:
Claude added per-iteration gradient norm logging to `main_v4.py` for all four methods, including a corrected gradient norm for SVRG (using the actual variance-reduced direction) and a manual norm computation inside SAGA's `saga_step()`. It also wrote `plot_grad_norms.py`, which produces an overlay plot, per-method grid, and rolling variance plot. The rolling variance plot was identified as the most theoretically significant visualization.

**How it was used**:
The logging code was integrated into the team's local `main_v4.py` and all four methods were rerun to generate updated result JSONs. The per-method and variance plots were included in the report. The team independently verified that the norm collapse for SVRG and SAGA matched the theoretical prediction of geometrically decreasing gradient variance.

---

## How Outputs Were Checked

- All generated code was run locally in the project virtual environment and outputs were inspected before use
- Plots were visually verified against expected theoretical behavior (e.g., SVRG/SAGA norm collapse, SGD noise floor persistence)
- Written descriptions of results in the report were rewritten by the team in their own words based on the actual plot outputs
- Mathematical content (update rules, convergence rates) was cross-checked against the original papers (Johnson & Zhang 2013, Defazio et al. 2014, Nguyen et al. 2017)
- No LLM-generated proofs or theoretical claims were used without verification against primary sources

---

## Responsibility Statement

The team takes full responsibility for the correctness, quality, and integrity of all code, experiments, written content, and conclusions presented in this project. LLM assistance was used to accelerate implementation and experiment design, but all outputs were reviewed, tested, and technically owned by the team members. Any errors in the final submission are the responsibility of the authors.

**Team members**:
- Noah Jacobson (Point of Contact)
- Kudzaishe Kadzimu
- Advaith Subramanian Sahasranamam