"""Microduck JUMP task — episodic ballistic vertical jump.

stand → crouch → both-feet takeoff → flight → land upright. The SCORED metric
(validity-gated apex in flight) is measured by the FROZEN ruler
``mjlab_microduck.ol_eval_jump``; this cfg is the outerloop AUTHOR's tunable
training recipe (rewards / curriculum / PPO config).

Forked from StandUp's config to inherit its proven groundcontact robot, domain
randomisation, 61-D observation contract, command slots, and nan_state
termination. Differences: a STAND init every episode, the minimal jump reward
set, the whole-robot ground-contact sensor the ruler needs, and a shorter
episode. See AGENTS.md ("Building a new env") and docs/roadmap.md.
"""
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    STAND_Z,
    make_microduck_standup_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

EPISODE_LENGTH_S = 4.0
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Rewards kept from StandUp's scaffold; everything else is stripped and replaced
# by the jump set below. body_ang_vel / angular_momentum / action_rate_l2 are
# the shared sim2real regularisers; self_collisions guards the groundcontact
# model; upright_linear gives a gradient toward vertical at every tilt.
_KEEP_REWARDS = {
    "action_rate_l2",
    "body_ang_vel",
    "angular_momentum",
    "self_collisions",
    "upright_linear",
}


def make_microduck_jump_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    # Inherit StandUp's fully-tuned scaffold, then reshape it for jumping.
    cfg = make_microduck_standup_env_cfg(play=play, rough=rough)

    # (1) Whole-robot ground-contact sensor: the SUPPORT/flight gate the ruler and
    #     the jump rewards read (found == 0 ⇒ no geom touches the terrain).
    robot_ground = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, robot_ground)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    # (2) Start STANDING every episode (jump is stand→…→stand, not a recovery).
    #     StandUp's set_ground_state supports a standing spawn; force it here.
    ev = cfg.events["set_ground_state"].params
    ev.update({
        "standing_prob": 1.0,
        "sitting_prob": 0.0,
        "face_down_prob": 0.0,
        "face_up_prob": 0.0,
    })
    # StandUp's init-mix curriculum would ramp the mix back toward prone poses —
    # remove it so the stand init sticks.
    cfg.curriculum.pop("ground_state_mix", None)

    # (3) Reward set: strip StandUp's standing recipe, keep the shared regularisers,
    #     add the minimal jump signal (the author expands from here).
    for name in list(cfg.rewards):
        if name not in _KEEP_REWARDS:
            del cfg.rewards[name]

    cfg.rewards["upright_linear"].weight = 0.5  # keep LIGHT: a jump needs launch, not a tilt-lock

    cfg.rewards["jump_airborne_height"] = RewardTermCfg(  # MAIN task signal
        func=microduck_mdp.jump_airborne_height,
        weight=20.0,
        params={"stand_z": STAND_Z, "ground_sensor": "robot_ground_contact"},
    )
    cfg.rewards["jump_takeoff_impulse"] = RewardTermCfg(  # bootstrap: attempt-to-jump gradient
        func=microduck_mdp.jump_takeoff_impulse,
        weight=2.0,
        params={"ground_sensor": "robot_ground_contact", "vz_cap": 3.0},
    )
    cfg.rewards["land_height"] = RewardTermCfg(  # return to a stand after the flight
        func=microduck_mdp.height_target_gaussian,
        weight=2.0,
        params={
            "std": 0.03,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["land_pose"] = RewardTermCfg(  # legs toward HOME on landing
        func=microduck_mdp.pose_l1_penalty,
        weight=1.0,
        params={"joint_indices": _LEG_JOINTS, "target_overrides": None},
    )

    # (4) Drop any StandUp reward-weight curriculum that targets a reward we removed
    #     (e.g. arrival_damping, height_stand_sharp, body_pose_tracking) — leaving it
    #     would set a weight on a missing reward. Curricula on kept rewards
    #     (action_rate_weight) and on commands/events (head/body ranges, pushes,
    #     CoM ranges) are harmless and stay.
    for cname in list(cfg.curriculum):
        term = cfg.curriculum[cname]
        if getattr(term, "func", None) is microduck_mdp.reward_weight:
            if term.params.get("reward_name") not in cfg.rewards:
                del cfg.curriculum[cname]

    # terminations inherited from StandUp: fell_over removed, nan_state kept —
    # exactly what jump needs (the robot must leave and re-touch the ground).
    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_jump",
    run_name="microduck_jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3000,
)
