# Contributing to Context Engine

Thank you for your interest in contributing! This guide will help you get set up and productive quickly.

## Development Environment Setup

### Prerequisites

- Python 3.10+
- A Cerebras API key (for running benchmarks only — not needed for engine development)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/context-engine.git
cd context-engine

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov
```

### Verify Your Setup

```bash
# Run the test suite
pytest

# Run with coverage
pytest --cov=engine --cov-report=term-missing
```

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_engine.py

# Specific test
pytest tests/test_engine.py::test_sha256_deterministic

# With verbose output
pytest -v
```

Some tests require `scikit-learn` and are skipped automatically if it's not installed.

## Running Benchmarks

Benchmarks require a Cerebras API key:

```bash
export CEREBRAS_API_KEY="your-key-here"

# Dense NIAH benchmark (30 turns, 5 needles)
python benchmarks/niah_dense.py

# Adversarial benchmark (shared keywords between needles and filler)
python benchmarks/niah_adversarial.py

# Goal-guided benchmark (fair, length-matched chunks)
python benchmarks/niah_goalguided.py

# 50-turn stress test
python benchmarks/benchmark_50turn.py
```

Results are saved to the `results/` directory as JSON and PNG charts.

## Project Structure

```
context-engine/
├── engine.py              # Core engine (ChunkLog, scorers, CerebrasSession)
├── agent.py               # Agent integration layer
├── demo_session.py         # Live demo script
├── requirements.txt        # Production dependencies
├── pyproject.toml          # Pytest configuration
├── tests/
│   └── test_engine.py      # Unit tests
├── benchmarks/
│   ├── niah_dense.py       # Dense needle-in-a-haystack
│   ├── niah_adversarial.py # Adversarial filler benchmark
│   ├── niah_goalguided.py  # Goal-guided scoring benchmark
│   ├── niah_boilerplate.py # Boilerplate detection benchmark
│   ├── niah_entity.py      # Entity extraction benchmark
│   ├── niah_semantic.py    # Semantic scoring benchmark
│   ├── niah_semantic_gap.py# Semantic gap analysis
│   └── benchmark_50turn.py # 50-turn stress test
└── results/                # Benchmark output (JSON + charts)
```

## How to Add a New Scorer

The engine uses a scorer interface pattern. Each scorer takes chunk text and context, and returns a priority score. Here's how to add one:

### 1. Define Your Scorer Class

Add your scorer to `engine.py` following the existing pattern:

```python
class MyScorer:
    """One-line description of what this scorer does."""

    def __init__(self, **params):
        # Initialize any state (models, config, etc.)
        pass

    def score(self, chunks: list[str], goal: str) -> list[float]:
        """Score a list of chunks relative to a goal.

        Args:
            chunks: List of chunk text strings.
            goal: The current user message / goal string.

        Returns:
            List of float scores, one per chunk.
            Higher scores = higher priority = retained longer.
        """
        scores = []
        for chunk in chunks:
            # Your scoring logic here
            score = 0.5  # Default neutral score
            scores.append(score)
        return scores
```

### 2. Integrate with ChunkLog

The `ChunkLog.compact()` method calls the scorer during compaction. Look at how `GoalGuidedScorer` is used in `ChunkLog.__init__()` and `ChunkLog.compact()` for the integration pattern.

### 3. Write Tests

Add tests to `tests/test_engine.py`:

```python
def test_my_scorer_basic():
    scorer = MyScorer()
    scores = scorer.score(["important bug fix"], "find the bug")
    assert scores[0] > 0.5  # Relevant content scores above baseline

def test_my_scorer_irrelevant():
    scorer = MyScorer()
    scores = scorer.score(["weather is nice today"], "find the bug")
    assert scores[0] <= 0.5  # Irrelevant content at or below baseline
```

### 4. Write a Benchmark

Create a new benchmark in `benchmarks/` following the existing pattern (see `niah_dense.py` for a minimal example). Benchmarks should:

- Run 10 sessions for statistical significance
- Use 8k-token context windows
- Inject 5 needles among filler turns
- Report average recall scores out of 5

### 5. Submit Your PR

- Include benchmark results in your PR description
- Ensure all existing tests still pass
- Add your scorer to the README's scorer comparison table

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Keep functions under 50 lines
- Prefer immutable patterns (return new objects, don't mutate)
- Add docstrings to public classes and functions

## Labels

We use these labels to categorize issues:

- `good-first-issue` — Great starting points for new contributors
- `scorer` — Related to scoring algorithms
- `benchmark` — Related to benchmarks and evaluation
- `integration` — External integrations and plugins

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-scorer`
3. Write tests first, then implementation
4. Run `pytest` and ensure all tests pass
5. Submit a pull request with benchmark results if applicable

## Questions?

Open an issue or start a discussion. We're happy to help!
