"""Bounded, exact, thread-owned device cache for PLE rows (opt-in).

Hot lookups transfer one hit/miss scalar to the host, not row IDs or row payload.
Misses fetch only unique missing IDs via the existing CPU/mmap reader. This is
NOT a fully asynchronous SSD lookup. Insertions materialize the bounded cache
state to avoid retaining an unbounded chain of scatter graphs. Each exact
64-bit row tag is stored as two uint32 words: MLX Metal does not support
scatter updates whose destination/value dtype is int64 or uint64.
"""
from __future__ import annotations
import threading
import numpy as np


def capacity_for(width: int, itemsize: int, byte_limit: int) -> int:
    if any(type(v) is not int or v<0 for v in (width,itemsize,byte_limit)) or not width or not itemsize:
        raise ValueError('Invalid device row cache dimensions/budget')
    n=byte_limit//(8+width*itemsize)
    return 1<<(n.bit_length()-1) if n else 0


def row_slots(ids,capacity,xp):
    # Fold the high half before using a 32-bit multiplicative hash; never use
    # slot equality as row equality. Both 32-bit tag words are compared to
    # preserve the full row ID, including IDs larger than 2**32.
    words=ids.astype(xp.uint64)
    folded=(words ^ (words >> 32)).astype(xp.uint32)
    return ((folded*xp.array(2654435761,dtype=xp.uint32)) & (capacity-1)).astype(xp.uint32)


def row_tag_words(ids, xp):
    """Lossless [low, high] uint32 tags; compatible with Metal scatter.

    This does not narrow the row ID to 32 bits. The original ID still goes
    through vocabulary validation; all-ones tags represent an empty slot.
    """
    words = ids.astype(xp.uint64)
    return xp.stack((words.astype(xp.uint32),
                     (words >> 32).astype(xp.uint32)), axis=-1)


class DeviceRowCache:
    def __init__(self,xp,width,dtype,row_count,*,byte_limit):
        self.xp=xp;self.width=int(width);self.dtype=dtype;self.row_count=int(row_count)
        bf16=getattr(xp,'bfloat16',None)
        if dtype == xp.float32:size=4
        elif dtype == xp.float16 or (bf16 is not None and dtype == bf16):size=2
        else:raise TypeError('Device PLE cache supports only F16, BF16 and F32 rows')
        if self.row_count < 1:raise ValueError('PLE vocabulary must be positive')
        self.capacity=capacity_for(self.width,size,byte_limit)
        self.row_bytes=8+self.width*size
        self._tags=self._values=None;self._owner=None;self._closed=False
        self.calls=self.all_hits=self.misses=self.missing_rows=0
        self._lock=threading.RLock()

    def lookup(self,indices,fetch):
        with self._lock:
            if self._closed:raise RuntimeError('Device PLE cache has been closed')
            owner=threading.get_ident()
            if self._owner is None:self._owner=owner
            if self._owner!=owner:raise RuntimeError('Device PLE cache used from a different MLX worker thread')
            xp=self.xp
            dtype=str(indices.dtype).split('.')[-1]
            if dtype not in ('int32','uint32','int64','uint64'):
                raise TypeError('PLE row indices must be integers')
            shape=tuple(indices.shape);flat=indices.reshape(-1)
            if not flat.size:return xp.zeros((*shape,self.width),dtype=self.dtype)
            if not self.capacity:return fetch(np.asarray(flat)).reshape(*shape,self.width)
            if self._tags is None:
                self._tags=xp.full((self.capacity,2),0xFFFFFFFF,dtype=xp.uint32)
                self._values=xp.zeros((self.capacity,self.width),dtype=self.dtype)
            slots=row_slots(flat,self.capacity,xp)
            valid=(flat>=0)&(flat<self.row_count)
            hit=xp.all(self._tags[slots]==row_tag_words(flat,xp),axis=-1)&valid
            self.calls+=1
            # One explicit scalar rendezvous remains. No false-hit heuristics.
            all_hit=bool(xp.all(hit).item())
            out=self._values[slots]
            if all_hit:
                self.all_hits+=1
                return out.reshape(*shape,self.width)
            self.misses+=1
            miss_positions=np.flatnonzero(~np.asarray(hit,dtype=bool))
            dev_positions=xp.array(miss_positions,dtype=xp.int64)
            missing=np.asarray(flat[dev_positions])
            if np.any(missing<0) or np.any(missing>=self.row_count):
                raise IndexError('PLE row index outside embedding vocabulary')
            unique,inverse=np.unique(missing.astype(np.int64),return_inverse=True)
            rows=fetch(unique)
            if tuple(rows.shape)!=(len(unique),self.width) or rows.dtype!=self.dtype:
                raise ValueError('PLE fallback returned wrong row shape/dtype')
            self.missing_rows+=len(unique)
            out[dev_positions]=rows[xp.array(inverse,dtype=xp.int64)]
            # Coalesce collisions before GPU scatter: duplicate scatter indices
            # have undefined winner order in MLX. Every output above is preserved.
            host_slots=np.asarray(row_slots(unique,self.capacity,np))
            winners={int(slot):i for i,slot in enumerate(host_slots)}
            positions=np.fromiter(winners.values(),dtype=np.int64)
            dst=xp.array(np.fromiter(winners.keys(),dtype=np.int64),dtype=xp.int64)
            pos=xp.array(positions,dtype=xp.int64)
            # MLX slicing yields a distinct array; explicit NumPy copy for tests.
            tags=self._tags.copy() if isinstance(self._tags,np.ndarray) else self._tags[:]
            values=self._values.copy() if isinstance(self._values,np.ndarray) else self._values[:]
            tags[dst]=row_tag_words(xp.array(unique[positions],dtype=xp.int64),xp)
            values[dst]=rows[pos]
            xp.eval(tags,values)
            self._tags,self._values=tags,values
            return out.reshape(*shape,self.width)

    def stats(self):
        with self._lock:
            return {'device_capacity':self.capacity,
                    'device_payload_bytes':self.capacity*self.row_bytes if self._tags is not None else 0,
                    'device_calls':self.calls,'device_all_hit_calls':self.all_hits,
                    'device_miss_calls':self.misses,'device_missing_rows':self.missing_rows,
                    'device_closed':self._closed}

    def close(self):
        with self._lock:
            self._tags=self._values=None;self._closed=True
