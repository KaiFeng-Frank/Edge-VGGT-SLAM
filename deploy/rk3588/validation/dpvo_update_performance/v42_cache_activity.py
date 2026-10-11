"""Small read-only activity query for the exact frozen v37 context cache.

Only replaces the v41 callback's full metrics() copy for NPU_used reporting.
Public metrics(), source rechecks, allocations, close and execution stay intact.
It returns the same last-history-then-live model flags as the original query;
the underlying SDK session sets npu_used at its actual run API call. No inference
or claim of numerical acceptance is made here.
"""

PARENT_CALLBACK_SHA256 = 'c7caef0c76673cd4511aa1a4c8a8a5427bfee691e1964227d782d0d6f26554eb'


def npu_used_by_model(cache):
    """Own just model-key/bool pairs under the original cache/session locks.

    This intentionally requires frozen BoundedContextCache internals. There is
    no guessed status or catch-and-fallback: a different backend contract fails.
    Lock order is cache then session, exactly as its metrics()/execute()/close().
    Nothing is changed, no array alias is returned, and a caller may freely
    mutate the returned dictionary. Lifecycle validation still uses full metrics.
    """
    with cache._lock:
        used = {row['model_key']: row['metrics'].get('npu_used') is True
                for row in cache._history}
        for key, session in cache._live.items():
            with session._lock:
                used[key] = session._metrics.get('npu_used') is True
        return used
