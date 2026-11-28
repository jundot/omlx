# SPDX-License-Identifier: Apache-2.0
"""The one admin seam a measurement or a delete reaches a Mac through.

The full run backend — stage, activate, supervise and the post-``ready``
observe measurement — belongs to the apply/start slice and is not part of this
change. What is shared already is the transport: every call to a Mac's own admin
API goes through the same two paths, so no route in this package invents a
second one.

* an empty ``target`` is *this* Mac: one loopback call that authenticates with
  the key this Mac already stores;
* any other Mac is reached through the existing SSH seam
  (``staging.run_remote_python``), whose bare interpreter authenticates its own
  loopback call. The origin never sees the key and pairing exchanges no new
  credential.

Two callers use it here:

* ``calibration`` runs one real completion per Mac during the measurement, and
  reads oMLX's own usage fields back as ``{"usage": …}``;
* the free-space delete relays one ``DELETE`` to the Mac that is short of disk.

Neither starts, stages or supervises anything.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from ..deployment import validate_ssh_target
from ..staging import DEFAULT_REMOTE_PYTHON, run_remote_python

# The per-Mac admin routes this module calls. They are oMLX's own routes; the
# assistant relays to them instead of re-implementing them.
MODEL_DELETE_PATH = "/admin/api/hf/models/{model_name}"

# The measurement's real-completion seam reads the timing fields oMLX itself
# computed for one streaming completion.
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
# Enough tokens that decode has more than one interval to measure, and short
# enough that a thinking model cannot spend minutes on a probe.
OBSERVE_MAX_TOKENS = 96
OBSERVE_TIMEOUT_SECONDS = 300.0
OBSERVE_PROMPT = (
    "Write one short sentence about why a quiet library is a good place to "
    "read. Reply with the sentence only."
)

_LOCAL_ADMIN_TIMEOUT = 30.0

logger = logging.getLogger(__name__)


class RemoteAdminError(RuntimeError):
    """The Mac answered, but not with a result."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"the Mac answered {status}: {detail}"[:500])


def _redacted(exc: BaseException) -> str:
    """A short failure message that cannot carry a key or a whole command."""

    text = str(exc).strip() or exc.__class__.__name__
    return text[-300:]


_REMOTE_ADMIN_SNIPPET = (
    "import json,sys\n"
    "import urllib.error,urllib.request\n"
    "from omlx.settings import get_settings,init_settings\n"
    "payload=json.loads(sys.argv[1])\n"
    "body=payload.get('body')\n"
    "data=json.dumps(body).encode('utf-8') if body is not None else None\n"
    "url='http://127.0.0.1:%d%s'%(int(payload['port']),payload['path'])\n"
    "request=urllib.request.Request(url,data=data,method=payload.get('method','POST'))\n"
    "request.add_header('Content-Type','application/json')\n"
    # This process is a bare ``python -c`` on the coordinator, not that Mac's
    # server, so nothing has loaded its settings yet. Reading the settings
    # object without that step raised "Settings not initialized" and made every
    # remote coordinator call fail; load them the documented way instead. The
    # key still never leaves the coordinator.
    "try:\n"
    "    key=getattr(get_settings().auth,'api_key',None)\n"
    "except RuntimeError:\n"
    "    key=getattr(init_settings().auth,'api_key',None)\n"
    "if key:\n"
    "    request.add_header('Authorization','Bearer '+key)\n"
    "try:\n"
    "    with urllib.request.urlopen(request,timeout=float(payload['timeout'])) as reply:\n"
    "        print(json.dumps({'ok':True,'status':getattr(reply,'status',200),"
    "'body':json.loads(reply.read().decode('utf-8') or '{}')}))\n"
    "except urllib.error.HTTPError as exc:\n"
    "    detail=exc.read().decode('utf-8','replace')\n"
    "    print(json.dumps({'ok':False,'status':exc.code,'detail':detail[:1000]}))\n"
    # A closed coordinator (or a refused loopback port) never answers, so it
    # cannot report an HTTP status. Without this the snippet died with a
    # traceback and the origin could only say "apply_failed" + diagnostics,
    # which is the wrong next step for "open oMLX on that Mac". Status 0 means
    # "no answer" and maps to retry_coordinator.
    "except (urllib.error.URLError,OSError) as exc:\n"
    "    print(json.dumps({'ok':False,'status':0,"
    "'detail':('the coordinator did not answer: '+str(exc))[:1000]}))\n"
)


def remote_admin(
    target: str,
    port: int,
    path: str,
    body: Mapping[str, Any] | None,
    *,
    method: str = "POST",
    timeout: float = 120.0,
) -> Mapping[str, Any]:
    """Call one admin route on the coordinator through the existing SSH seam.

    A local process on the coordinator authenticates its own loopback call with
    the key that Mac already stores; the origin never sees it and never stores
    it. That is why no new credential is exchanged by pairing.
    """

    if not 1 <= int(port) <= 65535:
        raise RemoteAdminError(502, "the coordinator's admin port is not known")
    reply = run_remote_python(
        validate_ssh_target(target),
        _REMOTE_ADMIN_SNIPPET,
        json.dumps(
            {
                "port": int(port),
                "path": path,
                "method": method,
                "body": None if body is None else dict(body),
                "timeout": float(timeout),
            },
            sort_keys=True,
        ),
        description=f"reach the coordinator's admin API at {path}",
        python_executable=DEFAULT_REMOTE_PYTHON,
        timeout=float(timeout) + 60.0,
    )
    if not isinstance(reply, Mapping):
        raise RemoteAdminError(502, "the coordinator answered with no JSON")
    if not reply.get("ok"):
        raw_status = reply.get("status")
        status = int(raw_status) if isinstance(raw_status, int) else 502
        raise RemoteAdminError(
            status,
            str(reply.get("detail") or "the coordinator refused the call"),
        )
    body_out = reply.get("body")
    return body_out if isinstance(body_out, Mapping) else {}


def delete_model_on_mac(
    *,
    port: int,
    model_name: str,
    target: str = "",
    timeout: float = 120.0,
) -> Mapping[str, Any]:
    """Delete one model on a measured Mac through the same admin seam.

    This is the free-space affordance's one write, and it is a relay: the card
    asks for a model it saw in that Mac's measured inventory, and the Mac's own
    ``DELETE /admin/api/hf/models/{name}`` does the deleting and the pool
    refresh. An empty ``target`` means this Mac (loopback with the key it
    stores); otherwise the coordinator story applies unchanged — a bare
    interpreter on that Mac authenticates its own loopback call, so the key
    never leaves the Mac and pairing exchanges no new credential.

    ``model_name`` is the folder name of the model on that Mac, which is what
    oMLX's route resolves. A name that Mac cannot resolve comes back as its own
    404, never as a silent success.
    """

    path = MODEL_DELETE_PATH.format(
        model_name=quote(str(model_name).strip(), safe="")
    )
    if target:
        return remote_admin(
            validate_ssh_target(target),
            port,
            path,
            None,
            method="DELETE",
            timeout=timeout,
        )
    try:
        return _local_admin(port, path, method="DELETE", timeout=timeout)
    except RemoteAdminError:
        raise
    except OSError as exc:  # a refused loopback port is "no answer", not a crash
        raise RemoteAdminError(0, f"this Mac did not answer: {_redacted(exc)}") from exc


def _local_admin(
    port: int,
    path: str,
    body: Mapping[str, Any] | None = None,
    *,
    method: str = "GET",
    timeout: float = _LOCAL_ADMIN_TIMEOUT,
) -> Any:
    """Call one admin route on this Mac's own server with the key it stores.

    The local coordinator *is* this process, so the same loopback contract the
    remote path uses applies here: the key never leaves the Mac and never
    crosses the assistant API. Callers run this off the event loop.
    """

    import urllib.error
    import urllib.request

    from ...settings import get_settings

    if not 1 <= int(port) <= 65535:
        raise RemoteAdminError(502, "the coordinator's admin port is not known")
    data = None if body is None else json.dumps(dict(body)).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}{path}", data=data, method=method
    )
    request.add_header("Content-Type", "application/json")
    try:
        key = getattr(get_settings().auth, "api_key", None)
    except RuntimeError:  # a bare interpreter without loaded settings
        key = None
    if key:
        request.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            raw = reply.read().decode("utf-8", "replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RemoteAdminError(exc.code, detail[:1000]) from exc




def _observe_body(model_id: str, *, stream: bool = False) -> dict[str, Any]:
    """One harmless completion whose own timing fields are the measurement."""

    body: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": OBSERVE_PROMPT}],
        "max_tokens": OBSERVE_MAX_TOKENS,
        "temperature": 0.0,
        "stream": stream,
    }
    if stream:
        # OpenAI-compatible servers put the final usage (and oMLX's TTFT /
        # prompt rate) in the last chunk only when usage is requested.
        body["stream_options"] = {"include_usage": True}
    return body


def _local_stream_observe(
    port: int, model_id: str = "", *, body: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Read the final usage chunk of one streaming completion on this Mac.

    The non-streaming path reports decode; TTFT and prefill live in the
    streaming chunk. This reads only oMLX's own usage fields and returns the
    last one seen; a missing chunk is ``{}`` (not a zero). ``body`` overrides
    the fixed observe request — the calibration runner sends its own prompt
    and token budget through the same loopback call and timeout.
    """

    import urllib.error
    import urllib.request

    from ...settings import get_settings

    if not 1 <= int(port) <= 65535:
        raise RemoteAdminError(502, "the coordinator's admin port is not known")
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}{CHAT_COMPLETIONS_PATH}",
        data=json.dumps(
            dict(body) if body is not None else _observe_body(model_id, stream=True)
        ).encode("utf-8"),
        method="POST",
    )
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "text/event-stream")
    try:
        key = getattr(get_settings().auth, "api_key", None)
    except RuntimeError:
        key = None
    if key:
        request.add_header("Authorization", "Bearer " + key)
    usage: Mapping[str, Any] = {}
    try:
        with urllib.request.urlopen(
            request, timeout=OBSERVE_TIMEOUT_SECONDS
        ) as reply:
            for raw in reply:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk, Mapping) and isinstance(
                    chunk.get("usage"), Mapping
                ):
                    usage = dict(chunk["usage"])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RemoteAdminError(exc.code, detail[:1000]) from exc
    return {"usage": usage}


_REMOTE_STREAM_OBSERVE_SNIPPET = (
    "import json,sys\n"
    "import urllib.error,urllib.request\n"
    "from omlx.settings import get_settings,init_settings\n"
    "payload=json.loads(sys.argv[1])\n"
    # An explicit body (the calibration runner's) wins; otherwise the fixed
    # observe request is built from the prompt/budget fields as before.
    "body=json.dumps(payload.get('body') or {'model':payload['model'],"
    "'messages':[{'role':'user','content':payload['prompt']}],"
    "'max_tokens':int(payload['max_tokens']),'temperature':0.0,"
    "'stream':True,'stream_options':{'include_usage':True}}).encode('utf-8')\n"
    "url='http://127.0.0.1:%d/v1/chat/completions'%int(payload['port'])\n"
    "request=urllib.request.Request(url,data=body,method='POST')\n"
    "request.add_header('Content-Type','application/json')\n"
    "request.add_header('Accept','text/event-stream')\n"
    "try:\n"
    "    key=getattr(get_settings().auth,'api_key',None)\n"
    "except RuntimeError:\n"
    "    key=getattr(init_settings().auth,'api_key',None)\n"
    "if key:\n"
    "    request.add_header('Authorization','Bearer '+key)\n"
    "usage={}\n"
    "try:\n"
    "    with urllib.request.urlopen(request,timeout=float(payload['timeout'])) as reply:\n"
    "        for raw in reply:\n"
    "            line=raw.decode('utf-8','replace').strip()\n"
    "            if not line.startswith('data:'):\n"
    "                continue\n"
    "            data=line[5:].strip()\n"
    "            if data=='[DONE]':\n"
    "                break\n"
    "            try:\n"
    "                chunk=json.loads(data)\n"
    "            except ValueError:\n"
    "                continue\n"
    "            if isinstance(chunk,dict) and isinstance(chunk.get('usage'),dict):\n"
    "                usage=chunk['usage']\n"
    "    print(json.dumps({'ok':True,'usage':usage}))\n"
    "except urllib.error.HTTPError as exc:\n"
    "    detail=exc.read().decode('utf-8','replace')\n"
    "    print(json.dumps({'ok':False,'status':exc.code,'detail':detail[:1000]}))\n"
    "except (urllib.error.URLError,OSError) as exc:\n"
    "    print(json.dumps({'ok':False,'status':0,"
    "'detail':('the coordinator did not answer: '+str(exc))[:1000]}))\n"
)


def remote_stream_observe(
    target: str, port: int, model_id: str = "", *, body: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Read one streaming completion's final usage on a remote coordinator.

    ``body`` overrides the fixed observe request when given: the calibration
    runner sends its own prompt and token budget through the same SSH seam,
    with the same timeout. The key is still read and used on the Mac itself.
    """

    if not 1 <= int(port) <= 65535:
        raise RemoteAdminError(502, "the coordinator's admin port is not known")
    payload: dict[str, Any] = {
        "port": int(port),
        "model": model_id,
        "prompt": OBSERVE_PROMPT,
        "max_tokens": OBSERVE_MAX_TOKENS,
        "timeout": OBSERVE_TIMEOUT_SECONDS,
    }
    if body is not None:
        payload["body"] = dict(body)
    reply = run_remote_python(
        validate_ssh_target(target),
        _REMOTE_STREAM_OBSERVE_SNIPPET,
        json.dumps(payload, sort_keys=True),
        description="measure TTFT/prefill on the coordinator",
        python_executable=DEFAULT_REMOTE_PYTHON,
        timeout=OBSERVE_TIMEOUT_SECONDS + 60.0,
    )
    if not isinstance(reply, Mapping):
        raise RemoteAdminError(502, "the coordinator answered with no JSON")
    if not reply.get("ok"):
        status = reply.get("status")
        raise RemoteAdminError(
            int(status) if isinstance(status, int) else 502,
            str(reply.get("detail") or "the coordinator refused the measurement"),
        )
    usage = reply.get("usage")
    return {"usage": dict(usage) if isinstance(usage, Mapping) else {}}


def stream_completion(
    *, port: int, body: Mapping[str, Any], target: str = ""
) -> Mapping[str, Any]:
    """One streaming completion on a measured Mac: body in, usage reply out.

    The per-node transport for ``calibration.calibrate_node``'s
    ``CompletionSeam`` contract: the same loopback/SSH seam the post-``ready``
    observe measurement uses, with the caller's completion body (produced by
    ``calibration.calibration_body``) instead of the fixed observe prompt. An
    empty ``target`` means this Mac — the in-process loopback call with the
    key it stores; any other Mac is reached through its own bare interpreter
    over the existing SSH seam, so no key leaves that Mac. The reply is the
    final usage chunk as ``{"usage": ...}``; a missing chunk is ``{}``, never
    a zero. The observe timeout bounds each completion.
    """

    if target:
        return remote_stream_observe(validate_ssh_target(target), port, body=body)
    return _local_stream_observe(port, body=body)
