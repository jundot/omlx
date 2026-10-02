# Adding a discovered Mac to an existing cluster

1. On the coordinator, open **Cluster > Add Mac**. Choose **Add this Mac** on the discovered Mac’s row.
2. On that new Mac, open oMLX > Cluster. Select the coordinator and choose **Show code**. If discovery is unavailable, use **Add by IP** with the coordinator address shown in the membership panel.
3. The coordinator keeps the selected Mac’s pairing form open while waiting for its request. Once the request arrives, enter the six-digit code shown on the new Mac and approve it. A request from another Mac does not enable this form.
4. After pairing, the new Mac appears as a candidate. Preview the new model split, review it, and explicitly apply it. Pairing alone does not change or reload the active model.

Discovery is not authorization: the existing code-based pairing and placement approval remain required. Invalid or expired codes show an error in the selected form; retry or cancel there. Adding a member still uses the existing model staging, compatibility checks and protected deployment reload path.
