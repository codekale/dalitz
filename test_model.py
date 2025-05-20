#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: kinematics.py
# Author: [Your Name]
# Date: [Current Date]
# Description: [Short description of the script's purpose]

from model import Model
from wave import Wave # type: ignore
from dynamic_amplitudes import breit_wigner

my_model = Model(
    name="MyModel",
    m_mother=5.28,
    m_1=0.45,
    m_2=0.135,
    m_3=0.135,
)

# Define two waves
wave_1 = Wave(
    name="rho(770)",
    m_0=0.775,
    width_0=0.149,
    m_1=0.13957,
    m_2=0.13957,
    s=1,
    isobar_system="23",
    dynamic_amplitude=breit_wigner,
)

my_model.add_wave(wave_1)

wave_2 = Wave(
    name="K*(892)0",
    m_0=0.892,
    width_0=0.050,
    m_1=0.493677,
    m_2=0.13957,
    s=1,
    isobar_system="12",
    dynamic_amplitude=breit_wigner,
)

my_model.add_wave(wave_2)

# Generate some pseudo-data
import jax.numpy as jnp

m_12 = jnp.array([1, 2, 3, 2])
m_23 = jnp.array([3, 2, 1, 2])

# Calculate the amplitude
amplitudes = my_model.calculate_amplitudes(m_12, m_23)
print("Amplitude:", amplitudes)
