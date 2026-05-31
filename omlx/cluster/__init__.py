# SPDX-License-Identifier: Apache-2.0
"""Request-level cluster router for multiple omlx servers.

A small stateless reverse proxy that spreads OpenAI-compatible requests across
several omlx backends (e.g. two Macs), routing each request to a server that
actually hosts the requested model and is least busy -- without GPU/memory-level
clustering. See ``router.py`` for the design and ``cluster.example.json`` for
config.
"""
