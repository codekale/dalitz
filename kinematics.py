#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: kinematics.py
# Author: [Your Name]
# Date: [Current Date]
# Description: [Short description of the script's purpose]

from jax import jit
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

    term1 = (m_mother**2 - (m_1 + m_2)**2)
    term2 = (m_mother**2 - (m_1 - m_2)**2)
    return jnp.sqrt(term1 * term2) / (2 * m_mother)

@jit
def bachelor_momentum_in_isobar_frame(m_123, m_12, m_3):
    """
    Calculate the momentum of the bachelor particle (3) in the rest frame of the (12) system

    Parameters:
    m_123 (float): Mass of the decaying particle.
    m_12 (float): Mass of the isobar.
    m_3: (float): Mass of the bachelor particle.

    Returns:
    float: Momentum of bachelor particle (3) in (12) rest frame.
    """

    E_3 = (m_123 - m_12**2 - m_3**2)/(2*m_12)
    return jnp.sqrt(E_3**2 - m_3**2)
