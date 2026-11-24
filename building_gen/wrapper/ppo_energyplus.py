# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from typing import Callable, Optional
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from building_gen.wrapper.bounded_distributions import LogisticNormal01
from building_gen.wrapper.utils import TrajectoryLogger
from building_gen.wrapper.spaces import NormalizeObservation, CustomRescaleAction


@dataclass
class PPOArgs:
    exp_name: str = "energyplus_ppo"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "Minergym_5archetypes"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_tags: list = None
    """wandb tags for this experiment"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    results_dir: str = "results"
    """the base directory for storing all results"""

    # Algorithm specific arguments
    env_id: str = "EnergyPlus-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    reward_type: str = "energy_comfort"
    """the type of reward function to use"""
    energy_weight: float = 1.0
    """the weight of the energy consumption penalty"""
    comfort_weight: float = 1.0
    """the weight of the comfort penalty"""
    stability_weight: float = 0.01
    """the weight of the stability penalty"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 96
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    eval_frequency: int = 100000
    """how often (in steps) to evaluate the policy during training"""
    eval_episodes: int = 3
    """number of episodes to evaluate during training"""
    episode_length_hours: int = 24
    """length of each episode in hours"""
    timesteps_per_hour: int = 4
    """number of timesteps per hour"""
    episode_log_frequency: int = 5
    """how often to log episode information"""
    detailed_rewards: bool = True
    """whether to log detailed reward components"""
    log_zone_analysis: bool = False
    """whether to log zone analysis"""
    log_efficiency_metrics: bool = True
    """whether to log efficiency metrics"""
    log_action_distribution: bool = True
    """whether to log action distribution"""
    log_value_function: bool = True
    """whether to log value function"""
    log_policy_stats: bool = True
    """whether to log policy statistics"""
    log_environment_stats: bool = True
    """whether to log environment statistics"""
    log_frequency: int = 5000
    """how often to log training information"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # EnergyPlus specific arguments
    building_file: str = None
    """the path to the EnergyPlus building file"""
    weather_file: str = None
    """the path to the EnergyPlus weather file"""
    weather_validation: str = None
    """the path to the EnergyPlus weather file for validation (optional)"""
    building_characteristics: dict = None
    """the characteristics of the building"""
    building_floor_area: float = None
    """the floor area of the building in m²"""
    eplus_output_dir: str = None
    """the directory for EnergyPlus output files"""

    def __post_init__(self):
        """Initialize computed values"""
        if self.wandb_tags is None:
            self.wandb_tags = []
        if self.building_characteristics is None:
            self.building_characteristics = {}


def make_env(
    env_id,
    path_to_building,
    path_to_weather,
    building_characteristics,
    reward_type,
    energy_weight,
    gamma,
    eplus_output_dir=None,
):
    def thunk():
        env = gym.make(
            env_id,
            path_to_building=path_to_building,
            path_to_weather=path_to_weather,
            building_characteristics=building_characteristics,
            reward_type=reward_type,
            energy_weight=energy_weight,
            eplus_output_dir=eplus_output_dir,
        )

        # The policy distribution already has support in [0, 1], matching the
        # normalized EnergyPlus action space.  Clipping here would make the
        # executed action disagree with the log probability used by PPO.
        norm_env = NormalizeObservation(env)
        return norm_env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        action_space = envs.single_action_space
        if not (np.allclose(action_space.low, 0.0) and np.allclose(action_space.high, 1.0)):
            raise ValueError(
                "PPO Agent requires a normalized Box action space with bounds [0, 1]; "
                f"received low={action_space.low}, high={action_space.high}"
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        # Debug: Log input to agent
        if hasattr(self, "debug_step") and self.debug_step % 5000 == 0:
            import logging

            logger = logging.getLogger("ppo")
            logger.info(f"=== AGENT INPUT DEBUG ===")
            logger.info(f"Agent input shape: {x.shape}")
            logger.info(f"Agent input bounds: [{x.min().item():.3f}, {x.max().item():.3f}]")
            logger.info(f"Agent input (first 10): {x.detach().cpu().numpy()[:10]}")
            logger.info(f"Agent input (last 10): {x.detach().cpu().numpy()[-10:]}")
            logger.info(f"=== END AGENT INPUT DEBUG ===")

        action_loc = self.actor_mean(x)
        # Keeping the latent standard deviation in a finite numerical range
        # prevents exp(logstd) from overflowing during an unstable update.
        action_logstd = self.actor_logstd.expand_as(action_loc).clamp(-20.0, 2.0)
        action_std = torch.exp(action_logstd)
        probs = LogisticNormal01(action_loc, action_std)
        if action is None:
            action = probs.deterministic_action if deterministic else probs.rsample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def main(args: PPOArgs):
    """
    Main function to run PPO algorithm.

    Args:
        args: PPOArgs object containing all training parameters
    """
    # Create directories for logs and outputs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = args.results_dir
    os.makedirs(run_dir, exist_ok=True)

    # Setup wandb if tracking is enabled
    if args.track:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
                tags=args.wandb_tags,
            )
        except ImportError:
            print("Warning: wandb not installed, tracking disabled")

    # Setup tensorboard
    writer = SummaryWriter(os.path.join(run_dir, "logs"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = logging.getLogger("ppo")

    # Set up detailed logging for debugging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(run_dir, "detailed_training.log"))],
    )

    # Ensure EnergyPlus interface logs are visible
    logging.getLogger("building_gen.wrapper.eplus_interface").setLevel(logging.INFO)
    logging.getLogger("building_gen.wrapper.spaces").setLevel(logging.INFO)
    logging.getLogger("building_gen.wrapper.rewards").setLevel(logging.INFO)
    logging.getLogger("building_gen.wrapper.energyplus_env").setLevel(logging.INFO)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    logger.info("Creating EnergyPlus environments...")
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=args.env_id,
                path_to_building=args.building_file,
                path_to_weather=args.weather_file,
                building_characteristics=args.building_characteristics,
                reward_type=args.reward_type,
                energy_weight=args.energy_weight,
                gamma=args.gamma,
                eplus_output_dir=(
                    args.eplus_output_dir if args.eplus_output_dir else os.path.join(run_dir, "eplus_outputs")
                ),
            )
            for _ in range(args.num_envs)
        ]
    )
    logger.info("EnergyPlus environments created successfully")
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    agent.debug_step = 0  # Add debug step counter
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    logger.info("Resetting environments to start training...")
    next_obs, _ = envs.reset(seed=args.seed)
    logger.info("Environment reset completed successfully")
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Use the evaluation frequency from args
    eval_frequency = args.eval_frequency
    next_eval_step = eval_frequency

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Update debug step counter
            agent.debug_step = global_step

            # Debug: Log action values
            if global_step % args.log_frequency == 0:
                logger.info(f"Step {global_step}: Raw NN output: {action.detach().cpu().numpy()}")
                logger.info(
                    f"Step {global_step}: Action space bounds: {envs.single_action_space.low} to {envs.single_action_space.high}"
                )
                logger.info(
                    f"Step {global_step}: Observation stats - min: {next_obs.min().item():.3f}, max: {next_obs.max().item():.3f}, mean: {next_obs.mean().item():.3f}"
                )
                logger.info(
                    f"Step {global_step}: Observation vector (first 10 values): {next_obs.detach().cpu().numpy()[:10]}"
                )
                logger.info(
                    f"Step {global_step}: Observation vector (last 10 values): {next_obs.detach().cpu().numpy()[-10:]}"
                )
                logger.info(f"Step {global_step}: Observation shape: {next_obs.shape}")

            # Execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Debug: Log reward and observation values
            if global_step % args.log_frequency == 0:
                logger.info(f"Step {global_step}: Reward: {reward[0]:.3f}")
                logger.info(
                    f"Step {global_step}: Next observation stats - min: {next_obs.min().item():.3f}, max: {next_obs.max().item():.3f}, mean: {next_obs.mean().item():.3f}"
                )
                logger.info(
                    f"Step {global_step}: Next observation vector (first 10 values): {next_obs.detach().cpu().numpy()[:10]}"
                )
                logger.info(
                    f"Step {global_step}: Next observation vector (last 10 values): {next_obs.detach().cpu().numpy()[-10:]}"
                )
                logger.info(f"Step {global_step}: Done: {next_done[0].item()}")

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # Check if it's time to evaluate the current policy
            if global_step >= next_eval_step:
                logger.info(f"Evaluating policy at step {global_step}")

                # Save current model to a temporary file
                temp_model_path = os.path.join(run_dir, f"temp_model_{global_step}.pt")
                torch.save(agent.state_dict(), temp_model_path)

                # Use validation weather if provided, else training weather
                validation_weather = args.weather_validation if args.weather_validation else args.weather_file

                # Evaluate the current policy
                eval_returns = ppo_evaluate(
                    model_path=temp_model_path,
                    make_env=make_env,
                    env_id=args.env_id,
                    reward_type=args.reward_type,
                    energy_weight=args.energy_weight,
                    path_to_building=args.building_file,
                    path_to_weather=validation_weather,
                    building_characteristics=args.building_characteristics,
                    eval_episodes=1,  # Just one episode for quick validation
                    run_name=f"{run_name}-validation-{global_step}",
                    device=device,
                    gamma=args.gamma,
                    save_trajectories=False,  # Don't save trajectories during validation
                    trajectories_dir=os.path.join(run_dir, "test_results"),
                    eplus_output_dir=(
                        args.eplus_output_dir if args.eplus_output_dir else os.path.join(run_dir, "eplus_outputs")
                    ),
                )

                # Log the validation return
                mean_return = np.mean(eval_returns)
                writer.add_scalar("validation/episodic_return", mean_return, global_step)

                # Remove the temporary model file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)

                # Set the next evaluation step
                next_eval_step = global_step + eval_frequency

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = os.path.join(run_dir, "model.pt")
        torch.save(agent.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Use validation weather if provided, else training weather
        validation_weather = args.weather_validation if args.weather_validation else args.weather_file

        episodic_returns = ppo_evaluate(
            model_path=model_path,
            make_env=make_env,
            reward_type=args.reward_type,
            energy_weight=args.energy_weight,
            env_id=args.env_id,
            path_to_building=args.building_file,
            path_to_weather=validation_weather,
            building_characteristics=args.building_characteristics,
            eval_episodes=1,
            run_name=f"{run_name}-eval",
            device=device,
            gamma=args.gamma,
            save_trajectories=True,  # Save trajectories for final evaluation
            trajectories_dir=os.path.join(run_dir, "test_results"),
            eplus_output_dir=args.eplus_output_dir if args.eplus_output_dir else os.path.join(run_dir, "eplus_outputs"),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
    if args.track:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass


def ppo_evaluate(
    model_path: str,
    make_env: Callable,
    reward_type: str,
    energy_weight: float,
    env_id: str,
    path_to_building: Optional[str],
    path_to_weather: Optional[str],
    building_characteristics: Optional[dict],
    eval_episodes: int,
    run_name: str,
    device: torch.device = torch.device("cpu"),
    gamma: float = 0.99,
    save_trajectories: bool = False,
    trajectories_dir: Optional[str] = None,
    eplus_output_dir: Optional[str] = None,
):
    """
    Evaluate a trained PPO model.

    Args:
        model_path: Path to the saved model
        make_env: Environment creation function
        env_id: Environment ID
        path_to_building: Path to building file
        path_to_weather: Path to weather file
        building_characteristics: Building characteristics dict
        eval_episodes: Number of episodes to evaluate
        run_name: Name for this evaluation run
        device: Device to run evaluation on
        gamma: Discount factor
        save_trajectories: Whether to save trajectories
        trajectories_dir: Directory to save trajectories (defaults to current directory)
        eplus_output_dir: Directory for EnergyPlus outputs
    """
    logger = logging.getLogger(__name__)

    # Create environment using the same setup as in training
    env = make_env(
        env_id,
        path_to_building,
        path_to_weather,
        building_characteristics,
        reward_type,
        energy_weight,
        gamma,
        eplus_output_dir,
    )()
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id,
                path_to_building,
                path_to_weather,
                building_characteristics,
                reward_type,
                energy_weight,
                gamma,
                eplus_output_dir,
            )
        ]
    )

    # Find both wrappers
    norm_wrapper = None
    rescale_wrapper = None
    temp_env = env

    while temp_env is not None:
        if isinstance(temp_env, NormalizeObservation):
            norm_wrapper = temp_env
        if isinstance(temp_env, CustomRescaleAction):
            rescale_wrapper = temp_env

        if hasattr(temp_env, "env"):
            temp_env = getattr(temp_env, "env")
        else:
            break

    # Create and load agent
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # Get the base environment to access its properties
    base_env = envs.envs[0].unwrapped
    observation_names = getattr(base_env, "observation_names", None)
    controlled_zones = getattr(base_env, "controlled_zones", None)
    uncontrolled_zones = getattr(base_env, "uncontrolled_zones", None)

    episodic_returns = []

    # Set up trajectories directory - use test_results naming
    if trajectories_dir is None:
        trajectories_base = "test_results"
    else:
        trajectories_base = trajectories_dir

    # Run evaluation for the specified number of episodes
    for episode in range(eval_episodes):
        # Initialize trajectory logger for this episode
        trajectory_logger = TrajectoryLogger(
            os.path.join(trajectories_base, f"episode_{episode}"), observation_names, logger=logger
        )

        # Reset environment
        obs, _ = envs.reset()
        done = False

        logger.info(f"Starting evaluation episode {episode+1}/{eval_episodes}")

        # Run episode until done
        while not done:
            with torch.no_grad():
                actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device), deterministic=True)
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminations, truncations, _ = envs.step(actions_np)

            # Denormalize observation if we have a normalization wrapper
            if norm_wrapper:
                denorm_obs = norm_wrapper.denormalize(obs[0])

                # Use the rescale wrapper if available, otherwise just use the raw actions
                if rescale_wrapper:
                    scaled_actions = rescale_wrapper.scale_action(actions_np[0])
                else:
                    scaled_actions = actions_np[0]

                # Log the denormalized observation in the trajectory
                trajectory_logger.log(denorm_obs, scaled_actions, rewards[0], controlled_zones, uncontrolled_zones)
            else:
                # If no normalization wrapper, log the observation as is
                trajectory_logger.log(obs[0], actions_np[0], rewards[0], controlled_zones, uncontrolled_zones)

            # Check if episode is done
            done = terminations[0] or truncations[0]
            obs = next_obs

        # Episode is done, get the total reward from the trajectory logger
        episode_return = trajectory_logger.total_reward
        episodic_returns.append(episode_return)

        logger.info(f"Evaluation episode {episode+1} completed: return={episode_return:.2f}")

        # Save the trajectory if requested
        if save_trajectories:
            trajectory_logger.save()
            logger.info(f"Saved trajectory for episode {episode+1}")

    # Calculate and log average return
    mean_return = sum(episodic_returns) / len(episodic_returns)
    logger.info(f"Evaluation completed. Mean return: {mean_return:.2f}")

    envs.close()
    return episodic_returns
