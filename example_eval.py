import argparse


def main():
    parser = argparse.ArgumentParser(description="DreamerV3")
    parser.add_argument(
        "-t", "--task", default="ABCD", type=str, help="Specify a task order"
    )
    # parser.add_argument("-r", "--runs", type=int, help="Specify a run number")
    args = parser.parse_args()

    import warnings
    import dreamerv3
    from dreamerv3 import embodied

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    task_order = "DCBA"
    run = 2
    input_type = "default"
    comment = "Noisy"

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["medium"])
    config = config.update(
        {
            "logdir": "/home/lagandua/projects/dreamerv3-calvin/logdir/eval_runs/",
            "run.train_ratio": 64,
            "run.log_every": 30,  # Seconds
            "batch_size": 16,
            "jax.prealloc": False,
            "encoder.mlp_keys": "vector",
            "decoder.mlp_keys": "vector",
            "encoder.cnn_keys": "image",
            "decoder.cnn_keys": "image",
            "run.from_checkpoint": f"/home/lagandua/projects/dreamerv3-calvin/logdir/{input_type}-{task_order}-Noisy-run{run}/checkpoint.ckpt",
            "jax.policy_devices": [0],
            "jax.train_devices": [0]
            # 'jax.platform': 'cpu',
        }
    )
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandBOutput(config.filter, logdir, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ],
    )

    #   import crafter
    from calvin import CalvinEnv
    from embodied.envs import from_gym

    #   env = crafter.Env()  # Replace this with your Gym env.
    from hydra import initialize, compose

    with initialize(config_path="./env_config/"):
        env_cfg = compose(config_name="calvin.yaml")
    env_cfg.task_order = task_order
    env = CalvinEnv(**env_cfg)
    env = from_gym.FromGym(env, obs_key="vector")  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    # replay = embodied.replay.Uniform(
    #     config.batch_length, config.replay_size, logdir / "replay"
    # )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    # embodied.run.train(agent, env, replay, logger, args)
    embodied.run.eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
