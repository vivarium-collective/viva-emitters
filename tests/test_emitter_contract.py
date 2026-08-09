import pytest

from viva_emitters.contract import EmitterContract
from viva_emitters.contracts import contract_for, register_contract


def test_contract_validates_output_kind():
    with pytest.raises(ValueError):
        EmitterContract(output_kind="weird")
    ok = EmitterContract(output_kind="zarr", output_uri_config_key="out_uri")
    assert ok.output_kind == "zarr"
    assert ok.output_uri_config_key == "out_uri"


def test_sqlite_emitter_self_describes():
    from viva_emitters.sqlite_emitter import SQLiteEmitter
    c = SQLiteEmitter.emitter_contract()
    assert c.output_kind == "sqlite"
    assert c.output_uri_config_key == "db_file"


def test_parquet_emitter_self_describes():
    from viva_emitters.parquet_emitter import ParquetEmitter
    c = ParquetEmitter.emitter_contract()
    assert c.output_kind == "parquet"
    assert c.output_uri_config_key  # whatever real key it uses


def test_xarray_emitter_self_describes():
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    from viva_emitters.xarray_emitter import XArrayEmitter
    c = XArrayEmitter.emitter_contract()
    assert c.output_kind == "zarr"
    assert c.output_uri_config_key == "out_uri"
    assert contract_for("xarray").output_kind == "zarr"
    assert contract_for("XArrayEmitter").output_kind == "zarr"
    assert contract_for(XArrayEmitter).output_kind == "zarr"


def test_contract_for_resolves_by_name_and_class():
    from viva_emitters.sqlite_emitter import SQLiteEmitter
    assert contract_for("sqlite").output_kind == "sqlite"
    assert contract_for("SQLiteEmitter").output_kind == "sqlite"
    assert contract_for(SQLiteEmitter).output_kind == "sqlite"


def test_ram_contract_without_editing_process_bigraph():
    c = contract_for("ram")
    assert c.output_kind == "ram"
    assert c.output_uri_config_key is None
    assert contract_for("RAMEmitter").output_kind == "ram"


def test_top_level_exports():
    import viva_emitters
    assert hasattr(viva_emitters, "EmitterContract")
    assert hasattr(viva_emitters, "contract_for")
    assert hasattr(viva_emitters, "register_contract")


def test_unknown_emitter_raises():
    with pytest.raises(KeyError):
        contract_for("does-not-exist")
