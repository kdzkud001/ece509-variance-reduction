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

### Prompt 6 — SAGA Conceptual Explanation and Pseudocode

**Date**: April 29, 2026

**Prompt**:
> "Please explain SAGA SGD method for SGD, and give me a pseudocode to solidify understanding."

**Output relied upon**:
Claude explained the core concepts of SAGA, including the gradient table, the unbiased variance-reduced update direction, and the convergence advantages over vanilla SGD. It provided a pseudocode outlining the initialization of the gradient table, the per-iteration sample, update, and table replacement steps.

**How it was used**:
The explanation was used to build foundational understanding of SAGA prior to implementation. The pseudocode served as a reference scaffold for writing the initial Python prototype.

---

### Prompt 7 — Adding Bias Terms to SAGA Prototype

**Date**: April 29, 2026

**Prompt**:
> "I have this SAGA prototype for logistic regression, and I would like to add biases to it. Please guide me through adding the derivative wrt the bias."

**Output relied upon**:
Claude identified and fixed a bug in the sigmoid function, suggested the correct form of the bias gradient (scalar derivative of the loss with respect to the bias term), and described how it could be incorporated into the `sample_gradient` function and tracked alongside the weight gradient in the SAGA update step.

**How it was used**:
The sigmoid bug fix was applied directly. The bias gradient derivation was reviewed against the logistic loss manually before being integrated into the prototype.

---

### Prompt 8 — Bug Confirmation in SAGA Implementation

**Date**: April 29, 2026

**Prompt**:
> "Confirm that these are the only fixes."

**Output relied upon**:
Claude identified three additional issues beyond those already flagged: a bug in the `grad_avg` update (incorrect incremental formula), a missing argument in a function call, and incorrect ordering of the `grad_avg` update relative to the table replacement step.

**How it was used**:
All three issues were located in the code, verified by the team, and corrected. The ordering fix for `grad_avg` was cross-checked against the original SAGA paper (Defazio et al. 2014) to confirm correctness.

---

### Prompt 9 — Test Case Generation for SAGA

**Date**: April 29, 2026

**Prompt**:
> "Please give me a sample test case."

**Output relied upon**:
Claude generated a test case using `make_classification` from scikit-learn to produce a synthetic binary classification dataset, and wrote a small script to run the SAGA implementation and verify convergence behavior.

**How it was used**:
The test script was run locally to confirm the implementation produced decreasing loss over iterations. Output was inspected visually before proceeding to real dataset evaluation.

---

### Prompt 10 — Visualizing Gradient Noise

**Date**: April 29, 2026

**Prompt**:
> "How can I visualize the noise created by the gradient?"

**Output relied upon**:
Claude advised plotting the gradient norm over training steps as a proxy for gradient noise, noting that variance-reduced methods should show a decaying norm compared to SGD's persistent noise floor.

**How it was used**:
This recommendation directly motivated Prompt 5, in which gradient norm logging was formally added to the full experimental pipeline.

---

### Prompt 11 — IEEE-Style Pseudocode for Report

**Date**: May 1, 2026

**Prompt**:
> "Given this code, please generate a pseudocode for an IEEE style paper. Remember to use small capitalizations and all other conventions where appropriate. Avoid ill-chosen variable names, ignore numpy by converting it into pseudocode. This is to be copy-pasted into MS word, into a one-cell table. Use 'Algorithm', 'Process', etc. for functions. Remember to also include the input and output for every function, and parse all super and subtext."

**Output relied upon**:
Claude produced IEEE-formatted pseudocode for the SAGA implementation, using small caps conventions, properly formatted sub- and superscripts, structured `Algorithm` and `Process` blocks with explicit **Input** and **Output** declarations, and variable names cleaned up from implementation-level names to notation appropriate for a formal paper.

**How it was used**:
The pseudocode was copied into a one-cell Word table in the report. Notation was reviewed by the team against the Defazio et al. 2014 paper and adjusted where variable naming diverged from the canonical presentation.

### Prompt 12 — SARAH Code Check

**Date**: May 3, 2026

**Prompt**:
Look at this code [SARAH code], is it properly written, and will it have similar behavior to SGD in some circumstances?

**Output relied upon**:
It gave two lines of code which were bugs before, "self.grads" and "v" needed to be replaced by "[p.clone() for p in self.params]" and "[vi.clone() for vi in v]" gave general suggestions that differences from SGD could be due to small batch sizes or mini-batches not being evaluated properly. 

**How it was used**:
This confirmed my suspicion based on the tests, that the SARAH code was not working properly and looking like SGD as a result. This issue was fixed the next day.


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
