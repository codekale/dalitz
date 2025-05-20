#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: model.py
# Author: [Your Name]
# Date: [Current Date]
# Description: [Short description of the script's purpose]

from jax import jit, vmap
import jax.numpy as jnp
from functools import partial

from angular_amplitudes import angular_amplitude_zemach
from kinematics import two_body_breakup_momentum, bachelor_momentum_in_isobar_frame
from barrier_factors import barrier_factor


class Model:
    def __init__(self, name: str, m_mother: float, m_1: float, m_2: float, m_3: float):
        self.name = name
        self.m_mother = m_mother
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3
        self.waves = []

    def add_wave(self, wave):
        """Add a wave to the model."""
        self.waves.append(wave)

    def get_waves(self):
        """Get the list of waves in the model."""
        return self.waves

    @partial(jit, static_argnums=(0,))
    def calculate_amplitudes(self, m_12, m_23):
        """
        Calculate the amplitudes for the model.

        Parameters:
        m_12 (float): The invariant mass of system 12.
        m_23 (float): The invariant mass of system 23.

        Returns:
        list: The calculated amplitudes.
        """

        def calculate_single_amplitude(m_12, m_23):
            # Calculate the invariant masse of the last system
            m_13 = jnp.sqrt(self.m_mother**2 + self.m_1**2 + self.m_2**2 - m_12**2 - m_23**2)    

            amplitudes = jnp.zeros((len(self.waves),), dtype=jnp.complex64)
            for i, wave in enumerate(self.waves):

                # Order the masses according to the isobar system
                if wave.isobar_system == "12":
                    m_1 = self.m_2
                    m_2 = self.m_1
                    m_3 = self.m_3
                    m_12_ = m_12
                    m_13_ = m_23
                    m_23_ = m_13
                
                elif wave.isobar_system == "13":
                    m_1 = self.m_1
                    m_2 = self.m_3
                    m_3 = self.m_2
                    m_12_ = m_13
                    m_13_ = m_12
                    m_23_ = m_23
                
                elif wave.isobar_system == "23":
                    m_1 = self.m_3
                    m_2 = self.m_2
                    m_3 = self.m_1
                    m_12_ = m_23
                    m_13_ = m_13
                    m_23_ = m_12

                # Calculate the dynamic amplitude
                dynamic_amplitude = wave.dynamic_amplitude(m=m_12,
                                                           m_1=wave.m_1,
                                                           m_2=wave.m_2,
                                                           s=wave.s,
                                                           m_0=wave.m_0,
                                                           width_0=wave.width_0)

                # Calculate the angular amplitude
                angular_amplitude = angular_amplitude_zemach(m_mother=self.m_mother,
                                                             m_1=m_1,
                                                             m_2=m_2,
                                                             m_3=m_3,
                                                             s=wave.s,
                                                             m_12=m_12_,
                                                             m_13=m_13_,
                                                             m_23=m_23_)
                # Barrier factor for mother decay
                q_0 = bachelor_momentum_in_isobar_frame(m_123=self.m_mother, m_12=wave.m_0, m_3=m_3)
                q = bachelor_momentum_in_isobar_frame(m_123=self.m_mother, m_12=m_12_, m_3=m_3)
                barrier_factor_mother = barrier_factor(L=wave.s, q_0=q_0, q=q, R=1)

                # Barrier factor for resonance decay
                q_0 = two_body_breakup_momentum(wave.m_0, m_1, m_2)
                q = two_body_breakup_momentum(m_12_, m_1, m_2)
                barrier_factor_resonance = barrier_factor(L=wave.s, q_0=q_0, q=q, R=1)

                # Calculate the full amplitude
                amplitudes = amplitudes.at[i].set(dynamic_amplitude * angular_amplitude * barrier_factor_mother * barrier_factor_resonance)
            return amplitudes

        # Vectorize the single amplitude calculation over the input arrays
        vectorized_calculation = vmap(calculate_single_amplitude, in_axes=(0, 0))
        return vectorized_calculation(m_12, m_23)
