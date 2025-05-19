#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: barrier_factors.py
# Author: [Your Name]
# Date: [Current Date]
# Description: [Short description of the script's purpose]

from jax import jit
import jax.numpy as jnp

@jit
def barrier_factor(L: int, q_0: float, q: float, R: float):
    """
    Calculate the barrier factor for a given angular momentum L, momenta q_0 and q, and radius R.

    Parameters:
    L (int): Relative angular momentum.
    q_0 (float): Breakup momentum of resonance.
    q (float): Breakup momentum of system.
    R (float): Meson radius of the resonance.

    Returns:
    float: Barrier factor.
    """

    if L < 0:
        raise ValueError("Angular momentum L must be non-negative.")
    
    # Calculate the barrier factor
    elif L == 0:
        return 1.0
    
    elif L == 1:
        return jnp.sqrt( (1+R**2*q_0**2) / (1+R**2*q**2) )

    elif L == 2:
        return jnp.sqrt( (9+3*R**2*q_0**2+R**4*q_0**4) / (9+3*R**2*q**2+R**4*q**4) )

    else:
        raise NotImplementedError("Barrier factor calculation for L > 2 is not implemented.")
