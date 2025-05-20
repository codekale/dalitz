#!/usr/bin/env python3
# type: ignore # -*- coding: utf-8 -*-
# File: wave.py
# Author: [Your Name]
# Date: [Current Date]
# Description: [Short description of the script's purpose]

class Wave:
    def __init__(self, name, m_0, width_0, m_1, m_2, s, isobar_system, dynamic_amplitude):
        """
        Initialize a Wave object.

        Parameters:
            name (str): The name of the wave.
            m_0 (float): The nominal mass of the resonance particle.
            width_0 (float): The nominal width of the resonance.
            m_1 (float): The mass of the first daughter particle.
            m_2 (float): The mass of the second daughter particle.
            s (int): The spin of the resonance.
            isobar_system (str): The isobar system associated with the wave.
            dynamic_amplitude (callable): A function or callable object representing the dynamic amplitude of the wave.
        """

        self.name = name
        self.m_0 = m_0
        self.width_0 = width_0
        self.m_1 = m_1
        self.m_2 = m_2
        self.s = s
        self.isobar_system = isobar_system
        self.dynamic_amplitude = dynamic_amplitude

    @property
    def name(self):
        """Get the name of the wave."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name of the wave."""
        self._name = value

    @property
    def m_0(self):
        """Get the nominal mass of the resonance particle."""
        return self._m_0

    @m_0.setter
    def m_0(self, value):
        """Set the nominal mass of the resonance particle."""
        self._m_0 = value

    @property
    def width_0(self):
        """Get the nominal width of the resonance."""
        return self._width_0

    @width_0.setter
    def width_0(self, value):
        """Set the nominal width of the resonance."""
        self._width_0 = value

    @property
    def m_1(self):
        """Get the mass of the first daughter particle."""
        return self._m_1

    @m_1.setter
    def m_1(self, value):
        """Set the mass of the first daughter particle."""
        self._m_1 = value

    @property
    def m_2(self):
        """Get the mass of the second daughter particle."""
        return self._m_2

    @m_2.setter
    def m_2(self, value):
        """Set the mass of the second daughter particle."""
        self._m_2 = value

    @property
    def s(self):
        """Get the spin of the resonance."""
        return self._s

    @s.setter
    def s(self, value):
        """Set the spin of the resonance."""
        self._s = value

    @property
    def isobar_system(self):
        """Get the isobar system associated with the wave."""
        return self._isobar_system

    @isobar_system.setter
    def isobar_system(self, value):
        """Set the isobar system associated with the wave."""
        self._isobar_system = value

    @property
    def dynamic_amplitude(self):
        """Get the dynamic amplitude of the wave."""
        return self._dynamic_amplitude

    @dynamic_amplitude.setter
    def dynamic_amplitude(self, value):
        """Set the dynamic amplitude of the wave."""
        self._dynamic_amplitude = value