# MouseLFP autoresearch

The autonomous research workflow is documented in [program.md](program.md). Read it and README before conducting model research.

- Use English for all project documentation, comments, prompts, log messages, and experiment descriptions.
- Keep the dataset, event eligibility, labels, raw window coordinates, 28-second duration, 14-second training stride, and day splits fixed.
- During autonomous experiments, primarily edit `train.py` and `models/*.py`. Preprocessing, models, optimizers, and training strategies are editable. Fit statistics on the training set only.
- `prepare.py`, `research_protocol.py`, `utils/mouse_lfp.py`, `utils/__init__.py`, `SETTINGS_MouseLFP.json`, `evaluate.py`, and `protocol.lock.json` form the fixed data and evaluation boundary during research.
- Use `./run_research.sh experiment` and compare mean validation accuracy across all seven mice within the same session. Establish a baseline before iterating and preserve complete keep/discard/crash records.
- `./run_research.sh restore` recovers the best training and model source code after backing up the current files. Protect existing user changes.
- Use separate smoke sessions for X15 and short budgets. After the search ends, `finalize` loads the selected weights for testing. Test results are excluded from model selection.
- Dependencies and Docker recipes are editable. Establish a matching baseline in a new session after changing the environment.
- Explicit engineering maintenance tasks may modify infrastructure while preserving the existing dataset, window, and split protocol.
- Start the research loop when the user requests experiments. Documentation changes and project checks do not start research.

See README for local execution commands and the available Docker environment.
