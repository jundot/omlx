# SPDX-License-Identifier: Apache-2.0
"""Client and service mailboxes exchange requests and replies through a daemon stand-in."""

from __future__ import annotations

import time

import pytest
from rdma_loopback import LoopbackLink, PythonWordOps

from omlx.cluster.rdma import layout
from omlx.cluster.rdma.mailbox import ClientMailbox, MailboxError, ServiceMailbox


@pytest.fixture
def link():
    loop = LoopbackLink(request_bytes=64 * 1024, reply_bytes=128 * 1024)
    yield loop
    loop.close()


def _attach(link):
    ops = PythonWordOps()
    client = ClientMailbox.attach(link.name, ops)
    service = ServiceMailbox.attach(
        link.name, link.socket_path, ops, mailbox_path=link.mailbox_path
    )
    return client, service


def test_request_and_reply_cross_the_link(link):
    client, service = _attach(link)
    try:
        assert client.connected
        assert client.sizes.max_request == 64 * 1024 - layout.CTRL
        seq = client.stage((b"hello ", b"world"))
        got = service.next_request(timeout_s=2)
        assert got is not None and got[0] == seq and bytes(got[1]) == b"hello world"
        service.reply(seq, (b"ack",))
        assert bytes(client.wait(seq, timeout_s=2)) == b"ack"
    finally:
        client.close()
        service.close()


def test_service_registers_in_poll_mode(link):
    client, service = _attach(link)
    try:
        client.stage((b"x",))
        assert service.next_request(timeout_s=2) is not None
        assert link.mode_lines == [b"MODE poll\n"]
    finally:
        client.close()
        service.close()


def test_a_second_service_is_refused_while_one_is_attached(link):
    client, service = _attach(link)
    try:
        with pytest.raises(MailboxError, match="refused the service: ERR busy"):
            ServiceMailbox.attach(
                link.name,
                link.socket_path,
                PythonWordOps(),
                mailbox_path=link.mailbox_path,
            )
        assert service.alive
    finally:
        client.close()
        service.close()


def test_a_daemon_ending_the_registration_is_noticed(link):
    client, service = _attach(link)
    try:
        assert service.alive
        link.say_bye()
        deadline = time.monotonic() + 2
        while service.alive and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not service.alive
    finally:
        client.close()
        service.close()


def test_a_request_left_by_a_previous_service_is_never_replayed(link):
    client = ClientMailbox.attach(link.name, PythonWordOps())
    try:
        link.wait_landed(client.stage((b"stale",)))
        service = ServiceMailbox.attach(
            link.name, link.socket_path, PythonWordOps(), mailbox_path=link.mailbox_path
        )
        try:
            assert service.next_request(timeout_s=0.2) is None
            fresh = client.stage((b"fresh",))
            got = service.next_request(timeout_s=2)
            assert got is not None and got[0] == fresh and bytes(got[1]) == b"fresh"
        finally:
            service.close()
    finally:
        client.close()


def test_payloads_larger_than_a_half_are_refused(link):
    client, service = _attach(link)
    try:
        with pytest.raises(MailboxError, match="larger than the mailbox half"):
            client.stage((b"\0" * client.sizes.request,))
        with pytest.raises(MailboxError, match="larger than the mailbox half"):
            service.reply(1, (b"\0" * service.sizes.reply,))
    finally:
        client.close()
        service.close()


def test_a_dropped_link_is_visible_to_both_ends(link):
    client, service = _attach(link)
    try:
        client.stage((b"x",))
        assert service.next_request(timeout_s=2) is not None
        assert service.alive
        link.drop()
        assert not client.connected
        assert not service.alive
    finally:
        client.close()
        service.close()


def test_missing_mailboxes_explain_themselves(tmp_path):
    with pytest.raises(MailboxError, match="is mcdma-rpcd running"):
        ClientMailbox.attach("nosuchlink", PythonWordOps())
    with pytest.raises(MailboxError, match="cannot open mailbox"):
        ServiceMailbox.attach(
            "nosuchlink",
            str(tmp_path / "s.sock"),
            PythonWordOps(),
            mailbox_path=str(tmp_path / "missing"),
        )
