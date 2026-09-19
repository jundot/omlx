# SPDX-License-Identifier: Apache-2.0
"""Canonical binding between a client-facing tool identity and its wire name.

The Responses layer has to expose flat function tools to a chat template while
remembering the client-facing identity behind each wire name: a namespace member
must come back as ``(namespace, name)``, and a call replayed from history must be
re-bound to the current request's wire name even if a collision changed it.

That bookkeeping used to live in anonymous ``{wire: (namespace, name)}`` dicts
with the join rule duplicated at each call site. This module is the single
representation both Responses emission sites and the namespace helpers share.
It deliberately does not unify the wire dialects of the other paths: Chat picks
``namespace::name`` and MCP picks ``server__tool``, and those are contractual.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def ensure_call_id(value: Optional[str]) -> str:
    """Return a non-empty tool-call correlation id.

    Every point that can synthesize one goes through here so a ``function_call``
    and its ``function_call_output`` can never disagree about which fallback
    they generated.
    """
    call_id = value.strip() if isinstance(value, str) else ""
    return call_id or f"call_{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class ToolBinding:
    """One tool as the model sees it, plus the identity the client sent."""

    wire_name: str
    name: str
    namespace: Optional[str] = None
    source: str = "function"
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    @property
    def identity(self) -> Tuple[Optional[str], str]:
        """The client-facing ``(namespace, name)`` this wire name stands for."""
        return (self.namespace, self.name)

    def to_chat_tool(self) -> Dict[str, Any]:
        """Render the tool in the nested Chat Completions shape templates read."""
        function: Dict[str, Any] = {"name": self.wire_name}
        if self.description:
            function["description"] = self.description
        if self.parameters:
            function["parameters"] = self.parameters
        if self.strict is not None:
            function["strict"] = self.strict
        return {"type": "function", "function": function}


@dataclass
class ToolBindingRegistry:
    """All bindings declared by one request, addressed by wire name or identity."""

    _by_wire: Dict[str, ToolBinding] = field(default_factory=dict)
    _by_identity: Dict[Tuple[Optional[str], str], str] = field(default_factory=dict)

    def _unique_wire_name(self, namespace: str, name: str) -> str:
        joined = f"{namespace.rstrip('_')}__{name.lstrip('_')}"
        wire = joined
        suffix = 2
        while wire in self._by_wire:
            wire = f"{joined}_{suffix}"
            suffix += 1
        return wire

    def register(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> ToolBinding:
        """Register a tool under a collision-free wire name and return it."""
        wire_name = (
            self._unique_wire_name(namespace, name) if namespace else name
        )
        binding = ToolBinding(
            wire_name=wire_name,
            name=name,
            namespace=namespace,
            source="namespace" if namespace else "function",
            description=description,
            parameters=parameters,
            strict=strict,
        )
        self._by_wire[wire_name] = binding
        self._by_identity[binding.identity] = wire_name
        return binding

    def resolve(self, wire_name: str) -> Tuple[Optional[str], str]:
        """Split a wire name back into ``(namespace, name)``.

        Names the model invented, and flat tools, pass through unchanged.
        """
        binding = self._by_wire.get(wire_name)
        if binding is None:
            return None, wire_name
        return binding.namespace, binding.name

    def resolve_identity(self, namespace: str, name: str) -> str:
        """Map a preserved ``(namespace, name)`` back to this request's wire name."""
        wire_name = self._by_identity.get((namespace, name))
        if wire_name is not None:
            return wire_name
        # The group is no longer declared, so there is no collision to avoid;
        # reuse the join rule so history stays readable to the model.
        return f"{namespace.rstrip('_')}__{name.lstrip('_')}"

    def apply_to_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Rewire namespace calls preserved in history to current wire names."""
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                namespace = function.pop("namespace", None)
                if namespace:
                    function["name"] = self.resolve_identity(
                        namespace, function["name"]
                    )

    def aliases(self) -> Dict[str, Tuple[Optional[str], str]]:
        """Wire-name map, for callers that still pass plain dicts around."""
        return {
            wire_name: binding.identity
            for wire_name, binding in self._by_wire.items()
            if binding.namespace
        }

    @property
    def wire_names(self) -> List[str]:
        return list(self._by_wire)

    def __len__(self) -> int:
        return len(self._by_wire)

    def __bool__(self) -> bool:
        return bool(self._by_wire)
