# SPDX-License-Identifier: Apache-2.0
"""omlx speculative-decoding wrappers.

This package collects integration code that bridges omlx scheduling/cache
infrastructure with upstream speculative-decoding implementations in mlx-lm
and mlx-vlm. Pure helpers (no business logic of their own) so the surface
of internal-API dependencies is easy to audit on each upstream bump.
"""
