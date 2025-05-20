#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: angular_amplitudes.py
# Author: [Your Name]
# Date: [Current Date]
# Description: [Short description of the script's purpose]

from jax import jit
import jax.numpy as jnp

@jit
def angular_amplitude_zemach(m_mother: float, m_1: float, m_2: float, m_3: float, s: int, m_12: float, m_13: float, m_23: float):
    """
    Calculate the angular amplitude.

    Parameters:
    m_mother (float): Mass of the decaying particle.
    m_1 (float): Mass of the first decay product.
    m_2 (float): Mass of the second decay product.
    m_3 (float): Mass of the third decay product.
    s (int): Spin of the resonance.
    m_12 (float): Invariant mass of system 12.
    m_13 (float): Invariant mass of system 13.
    m_23 (float): Invariant mass of system 23.

    Returns:
    float: Angular amplitude.
    """

    r = jnp.where(s == 0, 1.0, 0.0)
    r = jnp.where(s == 1, m_13**2 - m_23**2 - (m_mother**2 - m_3**2) * (m_1**2 - m_2**2) / m_12**2, r)
    r = jnp.where(s == 2, (m_23**2 - m_13**2 + (m_mother**2 - m_3**2) * (m_1**2 - m_2**2) / m_12**2)**2 \
                          - 1/3 * (m_12**2 - 2*m_mother**2 - 2*m_3**2 + ((m_mother**2 - m_3**2)**2) / m_12**2) \
                          * (m_12**2 - 2*m_1**2 - 2*m_2**2 + ((m_1**2 - m_2**2)**2) / m_12**2), r)
    return r
