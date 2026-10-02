# Forgetting an offline Mac or cluster

Each paired Mac in the active cluster list has a **Forget this Mac** action. It removes that Mac’s local pairing without contacting it. Model placements containing that Mac are removed after verified local teardown because a signed layer/rank assignment cannot remain valid after losing a member. Other Macs remain paired; create a new model placement with the remaining Macs.

The local Mac has a **Leave cluster** action. **Forget entire cluster** performs the same local departure: remove all saved placements and all local peer pairings. Neither action changes configuration on unreachable Macs. Each action requires its own confirmation.

Active requests and failed local process teardown block removal. Remote shutdown is explicitly unverified. SSH trust removal reuses the existing local pairing revocation path. Normal **Remove cluster setup** retains verified cluster-wide shutdown and does not remove pairings.
