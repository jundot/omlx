# SPDX-License-Identifier: Apache-2.0
"""Muse Spark on mlx-lm's model.

``model`` loads mlx-lm's Muse Spark for the VLM engine and cluster ranks, and
reads the layout that oMLX wrote before. ``oq_layout`` maps oQ's tensor names
to mlx-lm's layout and runs oQ's calibration through mlx-lm's decoder block.
"""
