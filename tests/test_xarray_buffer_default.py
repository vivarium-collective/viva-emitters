"""``transducer.buffer.size`` is optional and defaults to a sane, large value.

Regression guard for the "lost in translation" vendoring bug: a tiny buffer
(e.g. 3--4 emit steps) forces the ``XarrayBuffer`` to flush every few simulated
seconds, degrading latency and compression by ~2 orders of magnitude. The
library therefore requires no explicit ``buffer.size`` and, when omitted, uses
:py:data:`~viva_emitters.xarray_emitter.transducer.DEFAULT_BUFFER_SIZE` — sized
to flush only a handful of times per generation.
"""

import copy

import pytest

from viva_emitters.xarray_emitter.transducer import (
    DEFAULT_BUFFER_SIZE,
    XarrayTransducer,
)


def test_default_is_large_enough_to_avoid_pathological_flushing():
    # Flushing every few seconds is the bug we are guarding against; the default
    # should span many minutes of simulated time at a 1 Hz emission rate.
    assert DEFAULT_BUFFER_SIZE >= 100


def test_omitting_buffer_size_uses_default(minimal_xarray_config):
    config = copy.deepcopy(minimal_xarray_config)
    del config["transducer"]["buffer"]["size"]
    transducer = XarrayTransducer(config)
    assert transducer.buf_size == DEFAULT_BUFFER_SIZE


def test_omitting_buffer_key_entirely_uses_default(minimal_xarray_config):
    config = copy.deepcopy(minimal_xarray_config)
    del config["transducer"]["buffer"]
    transducer = XarrayTransducer(config)
    assert transducer.buf_size == DEFAULT_BUFFER_SIZE


def test_explicit_buffer_size_is_honored(minimal_xarray_config):
    config = copy.deepcopy(minimal_xarray_config)
    config["transducer"]["buffer"]["size"] = 512
    transducer = XarrayTransducer(config)
    assert transducer.buf_size == 512


@pytest.mark.parametrize("bad_size", [2, 0, -1, 1.5, "600"])
def test_invalid_explicit_buffer_size_still_rejected(minimal_xarray_config, bad_size):
    config = copy.deepcopy(minimal_xarray_config)
    config["transducer"]["buffer"]["size"] = bad_size
    with pytest.raises(TypeError):
        XarrayTransducer(config)
