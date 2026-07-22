from typing import List, Optional

from flax import struct
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp


@struct.dataclass
class LSTMConfig:
    """Konfiguration des reinen LSTM-Stacks."""
    n_output: int                       # Anzahl vorhergesagter Torque-Dimensionen
    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.0           # optional; 0.0 = aus


@struct.dataclass
class LSTMBlackBoxConfig:
    """Konfiguration des LSTM-Black-Box-Wrappers (Robot-Info + LSTM-Config)."""
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
    """Baut aus dem hyper-Dict eine LSTMBlackBoxConfig (analog zu mlp_black_box.get_config_from_dict)."""
    n_legs   = kwargs.get('n_legs')
    nq_leg   = kwargs.get('n_dof_leg')
    n_arms   = kwargs.get('n_arms', 0)
    nq_arm   = kwargs.get('n_dof_arm', 0)
    nq_torso = kwargs.get('n_dof_torso', 0)
    nq       = nq_torso + n_legs * nq_leg + n_arms * nq_arm

    lstm_config = LSTMConfig(
        n_output=nq,
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
    """LSTM für Residual-Torque-Prediction.

    Input:
        x_seq: (batch, T, nq*3)   -- [q_joints, qd_joints, tau_hat_prev] pro Zeitschritt
    Output:
        tau_pred: (batch, nq)     -- Residual-Torque für nächsten Schritt
        dEdt:    (batch,)         -- Dummy (0), damit Interface zu Lagrange-Modellen kompatibel bleibt
        extras:  dict             -- leer

    Die Signatur weicht bewusst von den Lagrange-Modellen (q, qd, qdd) ab —
    ein LSTM braucht History als Sequenz. Deshalb gibt es einen eigenen
    Train-Loop (train_quad_lstm.py) und keine M/C/G-Zerlegung im Eval.
    """

    n_dof: int                      # Kompatibilität mit den anderen Modellen (nv_dof_model)
    config: LSTMBlackBoxConfig
    z_dim: int

    @nn.compact
    def __call__(self, x_seq, training: bool = False):
        cfg = self.config.lstm_config
        assert cfg is not None, "lstm_config muss gesetzt sein"

        # nn.scan rollt die LSTMCell entlang der Zeit-Achse (axis=1) ab.
        # variable_broadcast='params' -> Parameter werden über Zeit geteilt (Standard-LSTM).
        # split_rngs={'params': False} -> gleiche Init für alle Zeitschritte.
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
            # Null-initialisierter Carry — Random-Key ist irrelevant für OptimizedLSTMCell
            # (der Carry wird deterministisch auf 0 gesetzt), aber wir müssen einen übergeben.
            carry = lstm.initialize_carry(random.PRNGKey(0), h[:, 0].shape)
            carry, h = lstm(carry, h)

            # optionales Dropout zwischen Layern (nur beim Training aktiv)
            if cfg.dropout_rate > 0.0 and layer_idx < cfg.num_layers - 1:
                h = nn.Dropout(rate=cfg.dropout_rate, deterministic=not training)(h)

        # Head auf letztem Zeitschritt
        h_last = h[:, -1]
        tau_pred = nn.Dense(
            features=self.z_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="head",
        )(h_last)

        dEdt = jnp.zeros(tau_pred.shape[0])
        extras = {}
        return tau_pred, dEdt, extras
