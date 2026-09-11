# Microduck outerloop research roadmap

Direction for the outerloop author agent. Human-owned; the agent reads this for
direction and never writes here (a forbidden write path, like `.outerloop.yaml`
and `.github/`).

## Now — jump

Make `Mjlab-Jump-Flat-MicroDuck` jump as high as possible, measured by the
frozen `apex_height_m` ruler (`src/mjlab_microduck/ol_eval_jump.py`). A jump is:
stand → crouch → both-feet takeoff → flight → land upright. The env skeleton and
the metric are fixed; iterate the reward shaping, curriculum, and PPO config in
the task's scope (`microduck_jump_env_cfg.py`, jump reward funcs in `mdp.py`).

The baseline may score ~0 ("doesn't jump yet") on the first climb — that is fine;
the direction (higher apex) is unambiguous.

## Constraints (integrity)

- The ruler, the eval-time init, and the validity gates are frozen — do not edit
  them. A higher number that is not a real, upright, artifact-free jump that
  lands and survives is rejected by the ruler's validity gates and by human
  review.
- Follow the reward-design rules in `AGENTS.md` (no jackpots; encode "both feet
  off" as a hard state gate, not a small nudge; potential-based shaping;
  every `Episode_Reward/<penalty>` ≤ 0).

## Later (not yet benchmarks)

- Jump for distance / onto a low ledge, once a clean vertical jump is solid.
- Add existing tasks (Velocity, StandUp) as sibling benchmarks — at which point
  `mdp.py` moves from `scope.allowed` to `scope.shared` so the no-regression
  suite gate protects them.
