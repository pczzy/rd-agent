from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import Env, QlibCondaConf, QlibCondaEnv, QTDockerEnv


class ModelCoSTEERSettings(CoSTEERSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_CoSTEER_")

    env_type: str = "conda"  # or "docker"
    """Environment to run model code in coder and runner: 'conda' for local conda env, 'docker' for Docker container"""


def get_qlib_env() -> Env:
    """Build the environment that runs qlib code, following MODEL_CoSTEER_ENV_TYPE.

    Shared by every qlib entry point (model coding, `qrun` backtest, feature validation)
    so that a single setting switches all of them between docker and conda.
    """
    conf = ModelCoSTEERSettings()
    if conf.env_type == "docker":
        return QTDockerEnv()
    if conf.env_type == "conda":
        return QlibCondaEnv(conf=QlibCondaConf())
    raise ValueError(f"Unknown env type: {conf.env_type}")


def get_model_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    env = get_qlib_env()
    env.conf.extra_volumes = {**env.conf.extra_volumes, **extra_volumes}
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


MODEL_COSTEER_SETTINGS = ModelCoSTEERSettings()
