# Pending pairing cleanup

Cancelling a join clears the local attempt immediately. If the coordinator has not acknowledged withdrawal, the dashboard keeps a cleanup warning and retries while the tab is visible. The saved cancellation proof survives restart; a new attempt to the same coordinator waits for cleanup, while other coordinators remain usable.

The current coordinator returns success when a pending request is already absent. For the older response reported in #3623, HTTP 404 is accepted only when its JSON body explicitly says `{"detail": "no pending join request"}`. This clears the saved withdrawal and permits a fresh attempt to that coordinator.

A generic 404, missing endpoint, invalid/oversized response, connection failure or server error does not prove cleanup. Those cases retain the warning and cancellation proof. This change does not make an unreachable coordinator acknowledge withdrawal and does not remove existing paired devices or model deployments.

To abandon an old cleanup locally, select **Forget this cleanup**, then **Confirm forget** in the warning. This permanently removes saved withdrawal records on this Mac, including after restart, without contacting the other Mac. The other Mac may retain the old request. Current pairing, an active join and model deployments stay unchanged. A later cancellation can create a new warning. If saving fails, the warning and cancellation proof remain.
