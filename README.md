# Bliss Attractors 🌀

A toy [Inspect](https://inspect.aisi.org.uk) implementation of the Bliss Attractor eval from [Claude 4 System Card Welfare Assessment](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf). It asks a model to talk to another instance of itself for a given number of turns, allowing it to reach its own peculiar equilibrium.

To replicate results from Anthropic's Model Card, run:

```bash
inspect eval tasks.py@self_interaction --model anthropic/claude-opus-4-20250514 --limit 1 --epochs 200 -T num_turns=30
```

It might be a lot of tokens tho!