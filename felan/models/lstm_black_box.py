from typing import List, Optional

from flax import struct
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp


@struct.dataclass
class LSTMConfig:
    n_output: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.0


@struct.dataclass
class LSTMBlackBoxConfig:
    # Robot
    nq: int
    n_legs: int
    nq_leg: int
    n_arms: int = 0
    nq_arm: int = 0
    nq_torso: int = 0
    # Sequenz
    time_window: int = 20
    # LSTM
    lstm_config: Optional[LSTMConfig] = None


def get_config_from_dict(kwargs):
    n_legs = kwargs.get('n_legs')
    nq_leg = kwargs.get('n_dof_leg')
    n_arms = kwargs.get('n_arms', 0)
    nq_arm = kwargs.get('n_dof_arm', 0)
    nq_torso = kwargs.get('n_dof_torso', 0)
    nq = nq_torso + n_legs * nq_leg + n_arms * nq_arm

    lstm_config = LSTMConfig(
        n_output=kwargs.get('n_output', nq),
        hidden_size=kwargs.get('lstm_hidden_size', 64),
        num_layers=kwargs.get('lstm_num_layers', 2),
        dropout_rate=kwargs.get('lstm_dropout', 0.0),
    )

    return LSTMBlackBoxConfig(
        nq=nq,
        n_legs=n_legs,
        nq_leg=nq_leg,
        n_arms=n_arms,
        nq_arm=nq_arm,
        nq_torso=nq_torso,
        time_window=kwargs.get('time_window', 20),
        lstm_config=lstm_config,
    )


class LSTMBlackBox(nn.Module):
    n_dof: int
    config: LSTMBlackBoxConfig

    @nn.compact
    def __call__(self, x_seq, training: bool = False):
        cfg = self.config.lstm_config
        assert cfg is not None, "lstm_config muss gesetzt sein"

        ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )

        h = x_seq
        for layer_idx in range(cfg.num_layers):
            lstm = ScanLSTM(cfg.hidden_size, name=f"lstm_layer_{layer_idx}")
            carry = lstm.initialize_carry(random.PRNGKey(0), h[:, 0].shape)
            carry, h = lstm(carry, h)

            if cfg.dropout_rate > 0.0 and layer_idx < cfg.num_layers - 1:
                h = nn.Dropout(rate=cfg.dropout_rate, deterministic=not training)(h)

        h_last = h[:, -1]
        tau_pred = nn.Dense(
            features=self.config.lstm_config.n_output,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="head",
        )(h_last)

        dEdt = jnp.zeros(tau_pred.shape[0])
        extras = {}
        return tau_pred, dEdt, extras
