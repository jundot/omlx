# SPDX-License-Identifier: Apache-2.0
"""Incremental Qwen XML arguments, confirmed by the native envelope parser.

The server admits only raw qwen3_coder engines and feeds fragments from the
existing envelope filter in FIFO order. A JSON object stays open until native
parsing confirms its entire emitted prefix; no tools execute in this module.
"""

import hashlib
import json
import uuid


class QwenArgumentStream:
    """One bounded envelope; completed calls are checked before closing JSON."""

    MAX_ENVELOPE_CHARS = 2 * 1024 * 1024
    MAX_HEADER_CHARS = 4096

    def __init__(self, tools):
        self.enabled = bool(tools)
        self.tools = tools or []
        self.buf = ""
        self.state = "seek"
        self.calls = []
        self.current = None
        self.param = ""
        self.config = {}
        self.raw = []
        self.string = False
        self.started = False
        self.first = True
        self.received = 0
        self.seen_parameters = set()
        self.duplicate_parameter = False

    def emit(self, argument="", name=None):
        f = {}
        # Native parsing keeps one value per parameter. Suppress repeated
        # JSON keys, then let finish reject any final value that differs
        # from the bytes already sent for that parameter.
        if name is None and self.duplicate_parameter:
            return {"index": len(self.calls) - 1, "function": f}
        if name is not None:
            f["name"] = name
        if argument:
            f["arguments"] = argument
            self.current["digest"].update(argument.encode("utf-8", "surrogatepass"))
            self.current["length"] += len(argument)
        tc = {"index": len(self.calls) - 1, "function": f}
        if name is not None:
            tc.update(id=self.current["id"], type="function")
        return tc

    def feed(self, text):
        if not self.enabled:
            return []
        from mlx_lm.tool_parsers.qwen3_coder import (
            _convert_param_value,
            _get_arguments_config,
            _string_types,
        )

        self.received += len(text)
        if self.received > self.MAX_ENVELOPE_CHARS:
            self.enabled = False
            self.buf = ""
            self.raw = []
            self.trailing_whitespace = []
            return []
        self.buf += text
        out = []
        while True:
            if self.state == "seek":
                i = self.buf.find("<tool_call>")
                if i < 0:
                    self.buf = self.buf[-10:]
                    break
                self.buf = self.buf[i + 11 :]
                self.state = "function"
            elif self.state == "function":
                self.buf = self.buf.lstrip()
                if len(self.buf) < 10:
                    break
                if not self.buf.startswith("<function="):
                    self.enabled = False
                    break
                i = self.buf.find(">")
                if i < 0:
                    if len(self.buf) > self.MAX_HEADER_CHARS:
                        self.enabled = False
                        self.buf = ""
                    break
                name = self.buf[10:i]
                if not any(
                    t.get("function", {}).get("name") == name for t in self.tools
                ):
                    self.enabled = False
                    break
                self.config = _get_arguments_config(name, self.tools)
                if not isinstance(self.config, dict) or any(
                    not isinstance(spec, dict) for spec in self.config.values()
                ):
                    # Boolean JSON schemas route through the final fallback;
                    # native conversion can fail even on a later parameter.
                    self.enabled = False
                    break
                self.seen_parameters = set()
                self.duplicate_parameter = False
                self.current = {
                    "name": name,
                    "id": "call_" + uuid.uuid4().hex[:8],
                    "digest": hashlib.sha256(),
                    "length": 0,
                }
                self.calls.append(self.current)
                out.append(self.emit("{", name))
                self.first = True
                self.buf = self.buf[i + 1 :]
                self.state = "parameter"
            elif self.state == "parameter":
                self.buf = self.buf.lstrip()
                if self.buf.startswith("</function>"):
                    self.buf = self.buf[11:]
                    self.state = "close"
                    continue
                if len(self.buf) < 11:
                    break
                if not self.buf.startswith("<parameter="):
                    self.enabled = False
                    break
                i = self.buf.find(">")
                if i < 0:
                    if len(self.buf) > self.MAX_HEADER_CHARS:
                        self.enabled = False
                        self.buf = ""
                    break
                from .tool_calling import _XML_PARAMETER_OPEN_RE
                # Native-only header spellings may disappear entirely during
                # EOF recovery. Stream only their common exact grammar.
                match = _XML_PARAMETER_OPEN_RE.fullmatch(self.buf[:i + 1])
                if match is None:
                    self.enabled = False
                    break
                self.param = match.group(1)
                self.duplicate_parameter = self.param in self.seen_parameters
                self.seen_parameters.add(self.param)
                self.buf = self.buf[i + 1 :]
                self.state = "value"
                self.raw = []
                self.trailing_whitespace = []
                self.started = False
                self.leading = True
                cfg = self.config.get(self.param, {})
                if self.param not in self.config or "type" not in cfg:
                    # Undeclared/untyped values can change type through final
                    # fallback coercion. Leave the remaining suffix to it.
                    self.enabled = False
                    break
                self.string = (
                    not cfg
                    or str(cfg.get("type", "string")).strip().lower() in _string_types
                )
                self.prefix = (
                    ("" if self.first else ", ")
                    + json.dumps(self.param, ensure_ascii=False)
                    + ": "
                )
                self.first = False
            elif self.state == "value":
                # Preserve a possible closing marker and one final newline.
                end = self.buf.find("</parameter>")
                closed = end >= 0
                if closed:
                    after = self.buf[end + 12:]
                    following = after.lstrip()
                    boundaries = ("<parameter=", "</function>")
                    if not any(following.startswith(marker) for marker in boundaries):
                        if (len(after) <= self.MAX_HEADER_CHARS and
                                any(marker.startswith(following) for marker in boundaries)):
                            break
                        # A literal close belongs to a different value in the
                        # recovery parser. Leave the JSON string open until
                        # final parsing settles that interpretation.
                        self.enabled = False
                        break
                n = end if closed else max(0, len(self.buf) - 13)
                if not n and not closed:
                    break
                part = self.buf[:n]
                self.buf = self.buf[n + (12 if closed else 0) :]
                if self.leading and part:
                    if part.startswith("\n"):
                        part = part[1:]
                    self.leading = False
                if closed and part.endswith("\n"):
                    part = part[:-1]
                if not self.string:
                    self.raw.append(part)
                    if closed:
                        try:
                            raw_value = "".join(self.raw)
                            value = _convert_param_value(
                                raw_value, self.param, self.config
                            )
                            from .tool_calling import _coerce_param_value
                            fallback = _coerce_param_value(
                                raw_value.strip(), self.param, self.config,
                                self.current["name"],
                            )
                            if json.dumps(value, ensure_ascii=False) != json.dumps(fallback, ensure_ascii=False):
                                self.enabled = False
                                break
                        except (ValueError, SyntaxError, TypeError):
                            self.enabled = False
                            break
                        out.append(
                            self.emit(
                                self.prefix + json.dumps(value, ensure_ascii=False)
                            )
                        )
                else:
                    prefix = ""
                    if not self.started:
                        self.raw.append(part)
                        pending = "".join(self.raw)
                        # Resolve split null/quote prefixes before deciding.
                        if not closed and len(pending) <= 4:
                            continue
                        # Recovery trims whitespace and decodes JSON strings.
                        # Three quotes cannot start one JSON string: the first
                        # two already close an empty string. Preserve such
                        # literal source prefixes, including Python docstrings.
                        if pending and (
                            pending[0].isspace()
                            or (pending[0] == '"' and not pending.startswith('"""'))
                        ):
                            self.enabled = False
                            break
                        if pending.lower().startswith("null") and pending[4:].isspace():
                            self.enabled = False
                            break
                        self.raw = []
                        if closed and pending.lower() == "null":
                            out.append(self.emit(self.prefix + "null"))
                            self.state = "parameter"
                            continue
                        prefix = self.prefix + '"'
                        part = pending
                        self.started = True
                    settled = part.rstrip()
                    if settled:
                        value = "".join(self.trailing_whitespace) + settled
                        self.trailing_whitespace = (
                            [part[len(settled) :]] if len(settled) < len(part) else []
                        )
                    else:
                        value = ""
                        if part:
                            self.trailing_whitespace.append(part)
                    if closed and self.trailing_whitespace:
                        # Preserve an open JSON string. Final parsing decides
                        # whether the trailing whitespace belongs to the value.
                        self.enabled = False
                    out.append(
                        self.emit(
                            prefix
                            + json.dumps(value, ensure_ascii=False)[1:-1]
                            + ('"' if closed and self.enabled else "")
                        )
                    )
                    if not self.enabled:
                        break
                if closed:
                    self.state = "parameter"
                elif not self.buf:
                    break
            elif self.state == "close":
                self.buf = self.buf.lstrip()
                if len(self.buf) < 12:
                    break
                if not self.buf.startswith("</tool_call>"):
                    self.enabled = False
                    break
                self.buf = self.buf[12:]
                self.state = "seek"
        return [x for x in out if x["function"]]

    def finish(self, tool_calls):
        if not self.calls:
            return None
        actual = tool_calls or []
        if len(actual) != len(self.calls):
            raise ValueError("Incremental tool count differs from native final parsing")
        out = []
        for i, (call, tc) in enumerate(zip(self.calls, actual)):
            length = call["length"]
            full = tc.function.arguments
            # Compare once at envelope completion. Retain only the digest and
            # character count while streaming, not a second argument body.
            digest = hashlib.sha256(full[:length].encode("utf-8", "surrogatepass"))
            if (
                tc.function.name != call["name"]
                or len(full) < length
                or digest.digest() != call["digest"].digest()
            ):
                raise ValueError(
                    "Incremental arguments differ from native final parsing"
                )
            out.append({"index": i, "function": {"arguments": full[length:]}})
        return out
