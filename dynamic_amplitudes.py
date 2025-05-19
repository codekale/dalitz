#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: dynamic_amplitudes.py
# Author: [Your Name]
# Date: [Current Date]
# Description: [Short description of the script's purpose]

from jax import jit
from kinematics import two_body_breakup_momentum
from barrier_factors import barrier_factor

@jit
def dynamic_width_for_breit_wigner(m: float, m_1:float, m_2:float, s:int, m_0:float, width_0:float):
    """
    Calculate the dynamic width for a Breit-Wigner resonance.

    Parameters:
    m (float): The mass of the particle.
    m_1 (float): The mass of the first decay product.
    m_2 (float): The mass of the second decay product.
    s (int): The spin of the resonance.
    m_0 (float): The nominal mass of the resonance.
    width_0 (float): The nominal width of the resonance.

    Returns:
    float: The dynamic width.
    """
    # Calculate the dynamic width using the Breit-Wigner formula
    q = two_body_breakup_momentum(m, m_1, m_2)
    q_0 = two_body_breakup_momentum(m_0, m_1, m_2)
    barrier_ratio = barrier_factor(L=s, q_0=q_0, q=q) / barrier_factor(L=s, q_0=q_0, q=q_0)
    width = width_0 * (m_0 / m) * (q / q_0)**(2 * s + 1) * barrier_ratio**2
    return width

@jit
def breit_wigner(m: float, m_1: float, m_2: float, s: int, m_0: float, width_0: float):
    """
    The amplitude of a Breit-Wigner resonance.

    Parameters:
    m (float): The mass of the particle.
    m_1 (float): The mass of the first decay product.
    m_2 (float): The mass of the second decay product.
    s (int): The spin of the resonance.
    m_0 (float): The nominal mass of the resonance.
    width_0 (float): The nominal width of the resonance.

    Returns:
    jax.numpy.complex64: amplitude.
    """
    import jax.numpy as jnp

    # Calculate the dynamic width
    width = dynamic_width_for_breit_wigner(m, m_1, m_2, s, m_0, width_0)
    
    # Calculate the Breit-Wigner amplitude
    amplitude = m_0 * width_0 / (m_0**2 - m**2 - 1j * m * width)
    return jnp.complex64(amplitude)
