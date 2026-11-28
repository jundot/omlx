# SPDX-License-Identifier: Apache-2.0
"""The Cluster Assistant's action vocabulary.

Every refusal this API returns carries exactly one next step (the design notes
the notes), and the whole product uses one fixed set of names for them. They
live here — not in a route — because ``plan``, ``recommend``, the free-space
delete and (later) the run backend all have to name the same step for the same
problem, and a second spelling of ``pair_mac`` would be a second product
behaviour.

The frozen twelve are ``measure | plan | apply | start | status | add_mac |
pair_mac | copy_model | configure_endpoint | retry_coordinator | diagnostics |
adjust_model_or_context``, plus the recommendation's four model-choice actions
(``choose_smaller_model``, ``choose_pipeline_model``, ``check_model_path``,
``free_disk_space``), which the recommendation module owns because only it can
name them.

``apply`` and ``start`` are part of the vocabulary before their routes exist:
they are what a client is told when the missing step is staging or launching,
and the card already renders them. The operation record that moves through
those states belongs to the run backend.
"""

from __future__ import annotations

ACTION_APPLY = "apply"
ACTION_START = "start"
ACTION_PLAN = "plan"
ACTION_MEASURE = "measure"
ACTION_STATUS = "status"
ACTION_ADD_MAC = "add_mac"
ACTION_PAIR_MAC = "pair_mac"
ACTION_COPY_MODEL = "copy_model"
#: Free space on a Mac before its stage can be copied there.
ACTION_FREE_DISK_SPACE = "free_disk_space"
ACTION_CONFIGURE_ENDPOINT = "configure_endpoint"
ACTION_RETRY_COORDINATOR = "retry_coordinator"
ACTION_DIAGNOSTICS = "diagnostics"

__all__ = [
    "ACTION_ADD_MAC",
    "ACTION_APPLY",
    "ACTION_CONFIGURE_ENDPOINT",
    "ACTION_COPY_MODEL",
    "ACTION_DIAGNOSTICS",
    "ACTION_FREE_DISK_SPACE",
    "ACTION_MEASURE",
    "ACTION_PAIR_MAC",
    "ACTION_PLAN",
    "ACTION_RETRY_COORDINATOR",
    "ACTION_START",
    "ACTION_STATUS",
]