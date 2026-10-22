import jax
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy.dynamics import dynamics_euler
import jax.numpy as jnp

params = load_params(Dynamics.so_rpy, "cf21B_500")

def f(t: float):
    return dynamics_euler(
        pos=jnp.array([0.0, 0.0, 0.0]),
        rpy=jnp.array([0.0, 0.0, 0.0]),
        vel=jnp.array([0.0, 0.0, 0.0]),
        rpy_rates=jnp.array([0.0, 0.0, 0.0]),
        cmd=jnp.array([0.0, 0.0, 0.0, t]),
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
    )
f_jit = jax.jit(f)

f_batched = jax.vmap(f_jit, in_axes=(0,), out_axes=(0, 0, 0, 0))

print(f_batched(jnp.linspace(0.45005, 0.4501, 10)))