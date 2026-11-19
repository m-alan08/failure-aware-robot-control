"""Stage H0 hardware software path (docs/HARDWARE_H0_IMPLEMENTATION.md).

Offline only. Nothing in this package talks to a robot, a camera or a network; the robot and
sensor interfaces are abstract and have no implementation. The controllers are the existing,
unchanged objects (Phase E estimator, latch, F1 model, Candidate C) composed in the exact order
of the confirmed simulation episode loops.
"""
