# SPDX-License-Identifier: Apache-2.0
"""
Store policy interface and default implementation for L1-to-L2 storage decisions.

The store policy makes two decisions after data is written to L1:
1. Which L2 adapter(s) should each key be stored to?
2. After a successful L2 store, should the key be deleted from L1?
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)


@dataclass(frozen=True)
class AdapterDescriptor:
    """
    Lightweight descriptor for an L2 adapter, giving the store policy
    enough information to distinguish adapters without exposing runtime
    objects.
    """

    index: int
    """Position in the L2 adapters list."""

    config: L2AdapterConfigBase
    """The adapter's configuration object."""

    @property
    def type_name(self) -> str:
        """
        Registered adapter type name (e.g., "mock", "disk", "redis").

        Derived from the config's registered type via reverse lookup.

        Returns:
            str: The registered type name.
        """
        return get_type_name_for_config(self.config)


class StorePolicy(ABC):
    """
    Abstract interface for store decisions.

    The store policy is called by the StoreController to decide:
    1. Which adapter(s) to store each key to (select_store_targets).
    2. Which keys to delete from L1 after successful L2 store
       (select_l1_deletions).
    """

    @abstractmethod
    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Decide which keys to store to which L2 adapters.

        Args:
            keys: Keys that were just written to L1 and are
                candidates for L2 storage.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to list of keys to store
            to that adapter. Keys absent from all lists are NOT
            stored to L2.
        """

    @abstractmethod
    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Decide which keys to delete from L1 after successful L2 store.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            Keys to delete from L1. Empty list means keep all.
        """


# -----------------------------------------------------------------------------
# Registry: store policy name -> policy class
# -----------------------------------------------------------------------------

_STORE_POLICY_REGISTRY: dict[str, type[StorePolicy]] = {}


def register_store_policy(
    name: str,
    policy_cls: type[StorePolicy],
) -> None:
    """
    Register a store policy class under a name.

    Each policy module should call this at import time.

    Args:
        name: Policy name (e.g. "default").
        policy_cls: A concrete StorePolicy subclass.
    """
    if name in _STORE_POLICY_REGISTRY:
        raise ValueError(f"Store policy already registered: {name!r}")
    _STORE_POLICY_REGISTRY[name] = policy_cls


def get_registered_store_policies() -> list[str]:
    """Return the list of registered store policy names."""
    return list(_STORE_POLICY_REGISTRY)


def create_store_policy(name: str) -> StorePolicy:
    """
    Create a store policy instance by name.

    Args:
        name: Registered policy name.

    Returns:
        A new StorePolicy instance.

    Raises:
        ValueError: If no policy is registered under the given name.
    """
    if name not in _STORE_POLICY_REGISTRY:
        known = ", ".join(sorted(_STORE_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown store policy {name!r}. Known: {known}")
    return _STORE_POLICY_REGISTRY[name]()


class DefaultStorePolicy(StorePolicy):
    """
    Default store policy: store all keys to all adapters,
    never delete from L1.
    """

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Store all keys to all adapters.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from every adapter index to the full list of keys.
        """
        return {ad.index: list(keys) for ad in adapters}

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Never delete from L1.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            Empty list (keep all keys in L1).
        """
        return []


class BufferOnlyStorePolicy(DefaultStorePolicy):
    """
    Buffer-only store policy: store all keys to all adapters,
    then delete them from L1 immediately.

    Use this with NoOpEvictionPolicy to avoid useless LRU
    tracking overhead when L1 is a pure write buffer.

    Inherits ``select_store_targets`` from ``DefaultStorePolicy``
    (store all keys to all adapters) and only overrides the L1
    deletion decision.
    """

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Delete all keys from L1 after successful L2 store.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            All keys (remove everything from L1).
        """
        return list(keys)


register_store_policy("default", DefaultStorePolicy)
register_store_policy("skip_l1", BufferOnlyStorePolicy)


class StripedStorePolicy(StorePolicy):
    """Striped store policy: distribute keys across adapters by hash.

    Each key is assigned to exactly one adapter via an MD5-based
    hash of the key, i.e. ``md5(str(key)) % len(adapters)``, spreading write (and read) pressure evenly across
    multiple SSDs.  Unlike :class:`DefaultStorePolicy` which mirrors
    every key to every adapter, this policy stores each key only once,
    sacrificing redundancy for throughput and capacity.

    Pair with multiple ``--l2-adapter`` instances pointing to different
    SSD paths and ``--l2-store-policy striped`` to enable.

    The matching prefetch policy (:class:`DefaultPrefetchPolicy`) already
    picks the first adapter that has a key, so no prefetch-side change
    is needed — under striped storage exactly one adapter has any given
    key.
    """

    @staticmethod
    def _adapter_index_for_key(
        key: ObjectKey,
        num_adapters: int,
    ) -> int:
        """Deterministically pick an adapter index for *key*.

        Uses MD5 on the key's string representation for a deterministic,
        cross-process-stable hash.  Python's built-in ``hash()`` is
        randomized per process via ``PYTHONHASHSEED`` (Python 3.3+), so
        it would route the same key to different adapters after a server
        restart — orphaning persistent L2 data.

        MD5 is sufficient here: we need uniform distribution, not
        cryptographic security, and it avoids the ``blake3`` dependency.

        Args:
            key: The object key to route.
            num_adapters: Number of available adapters.

        Returns:
            Adapter index in ``[0, num_adapters)``.
        """
        h = hashlib.md5(str(key).encode())
        return int.from_bytes(h.digest()[:8], "big") % num_adapters

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """Assign each key to exactly one adapter via hash-based striping.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to the list of keys assigned to
            that adapter. Each key appears in exactly one adapter's
            list. If ``adapters`` is empty, returns an empty dict (no
            L2 storage).
        """
        if not adapters:
            return {}

        num_adapters = len(adapters)
        # Pre-sort by index for deterministic modulo mapping
        sorted_adapters = sorted(adapters, key=lambda a: a.index)
        result: dict[int, list[ObjectKey]] = {ad.index: [] for ad in sorted_adapters}

        for key in keys:
            slot = self._adapter_index_for_key(key, num_adapters)
            adapter_id = sorted_adapters[slot].index
            result[adapter_id].append(key)

        return result

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """Never delete from L1 (same as DefaultStorePolicy).

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            Empty list (keep all keys in L1).
        """
        return []


register_store_policy("striped", StripedStorePolicy)
