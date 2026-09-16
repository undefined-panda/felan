from flax import struct
from flax import linen as nn
from jax import random
import jax.numpy as jnp

@struct.dataclass
class LSTMBlackBoxConfig:
    n_output: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.0
    time_window: int = 20 # sequence length

def get_config_from_dict(kwargs):
    return LSTMBlackBoxConfig(
        n_output=kwargs.get('n_output'),
        hidden_size=kwargs.get('lstm_hidden_size', 64),
        num_layers=kwargs.get('lstm_num_layers', 2),
        dropout_rate=kwargs.get('lstm_dropout', 0.0),
    )

class LSTMBlackBox(nn.Module):
    n_dof: int
    config: LSTMBlackBoxConfig

    @nn.compact
    # q, qd, qdd are added to match call of other models in train pipeline
    def __call__(self, q, qd, qdd, history, training: bool = False):
        ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell, variable_broadcast="params",
            split_rngs={"params": False}, in_axes=1, out_axes=1,
        )

        for layer_idx in range(self.config.num_layers):
            lstm = ScanLSTM(self.config.hidden_size, name=f"lstm_layer_{layer_idx}")
            carry = lstm.initialize_carry(random.PRNGKey(0), history[:, 0].shape)
            carry, history = lstm(carry, history)

            if self.config.dropout_rate > 0.0 and layer_idx < self.config.num_layers - 1:
                history = nn.Dropout(rate=self.config.dropout_rate, deterministic=not training)(history)

        x = nn.Dense(
            features=self.config.n_output,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="head")(history[:, -1])
        dEdt = jnp.zeros(x.shape[0])
        extras = {}
        return x, dEdt, extras
