from jax import jit

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: kinematics.py
# Author: [Your Name]
# Date: [Current Date]
# Description: [Short description of the script's purpose]

import jax.numpy as jnp

@jit
def two_body_breakup_momentum(m_mother: float, m_1: float, m_2: float):
    """
    Calculate the two-body breakup momentum in the rest frame of the decaying particle using JAX for speedup.

    Parameters:
    m_mother (float): Mass of the decaying particle.
    m_1 (float): Mass of the first decay product.
    m_2: (float): Mass of the second decay product.

    Returns:
    float: Breakup momentum.
    """
    if m_mother < m_1 + m_2:
        raise ValueError("The decaying particle's mass must be greater than the sum of the decay products' masses.")

    term1 = (m_mother**2 - (m_1 + m_2)**2)
    term2 = (m_mother**2 - (m_1 - m_2)**2)
    return jnp.sqrt(term1 * term2) / (2 * m_mother)