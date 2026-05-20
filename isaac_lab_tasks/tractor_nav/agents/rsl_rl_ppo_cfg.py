"""PPO config for Isaac-Tractor-Nav-v0 using rsl_rl.

NOTE: stock rsl_rl ActorCritic is MLP-only. The dict observation
{image, proprio} needs a CNN+MLP actor-critic. The cleanest path is to
register a custom ActorCriticCNN class — see actor_critic_cnn.py.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TractorNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "tractor_nav"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        # If you wire a custom CNN class in rsl_rl, set class_name here.
        # Default MLP will fail on the dict obs — see note above.
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
