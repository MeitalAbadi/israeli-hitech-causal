# How War and AI Reshaped Israeli Hi-Tech Labor (2022 to 2026)

A causal-inference look at what actually happened to Israeli hi-tech jobs after October 7, 2023, and after generative-AI tools became part of every developer's daily workflow.

I'm building this in the open over a few intensive days. The point isn't a polished academic paper. It's to ask one specific question with the data we actually have, and answer it as honestly as the data allows.

## The question

How did two simultaneous shocks reshape the Israeli hi-tech labor market between late 2022 and 2026?

1. The Iron Swords war that began on October 7, 2023.
2. The rapid integration of generative-AI tools into developer workflows (ChatGPT in late 2022, GPT-4 in early 2023, Claude 3.5 and Devin in 2024).

Almost everyone in the Israeli tech industry has a theory about how these two forces have changed the job market. Almost no one has run the numbers properly with causal methods. That's the gap I want to close, even if only partially.

## Why I'm building this

I'm finishing my data-science studies in Israel and looking for a hi-tech role right now. Every conversation I have with people in the industry circles around these two forces: funding rounds drying up or recovering, layoffs, hiring freezes, sudden re-orgs, juniors getting squeezed out. I wanted a project that took my own field's biggest current question and applied a real causal toolkit to it, instead of guessing like everyone else.

## Data sources (all public)

- Bank of Israel labor force statistics
- Central Bureau of Statistics (CBS) employment data
- StartupNation Central public reports
- Crunchbase (free tier)
- Tel Aviv Stock Exchange (TASE) public market data
- A hand-compiled timeline of major events: the war, AI tool releases, large reported Israeli tech layoffs

No paid data sources, no scraped private data, no LinkedIn premium. Anyone with a laptop and a coffee can rebuild this.

## Methods

Two main causal frameworks:

1. **Interrupted Time Series (ITS)** for the war shock, anchored to October 7, 2023. Pre-trend modeled on the 2 years before, post-trend on everything after.
2. **Difference-in-Differences (DiD)** comparing AI-exposed roles (junior dev, QA, BI) to less-exposed roles (product, design, sales), anchored to the GPT-4 release in March 2023.

As a robustness check, **Synthetic Control** comparing Israeli hi-tech to a weighted combination of comparable ecosystems (Singapore, Estonia, Berlin, UK).

Implementation uses CausalPy (PyMC-backed Bayesian causal inference) so the output isn't just a point estimate. We get a posterior distribution over the causal effect, with the uncertainty made explicit.

## Project structure

```
data/raw/        # source files as downloaded, never modified
data/processed/  # cleaned and merged, ready for modeling
notebooks/       # exploratory work, EDA, first-pass modeling
src/             # importable utilities and pipelines
results/plots/   # final figures
docs/            # methodology notes and decisions
```

## Status

In progress. Started as an intensive build over a couple of days. Will keep refining.

Current focus: data acquisition + cleaning + first ITS run.

## What this is not

A final answer. Causal inference on observational data is hard, especially when two big shocks land within twelve months of each other and partially confound each other. The point of this project is to be honest about what the data can and cannot tell us, and to make every assumption explicit. Where the answer is "we cannot tell from this data," I plan to say so.

## License

MIT. Use what's useful, just credit me.
