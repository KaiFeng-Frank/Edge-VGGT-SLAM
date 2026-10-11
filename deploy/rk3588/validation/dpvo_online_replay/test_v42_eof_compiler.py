"""Fake SDK compile-operation tests; no toolkit/model/ORT/board execution."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def compiler():
    path = Path(os.environ.get("V42_COMPILER", HERE / "compile-v42-m16-eof-source.py"))
    if not path.is_file():
        path = HERE.parent / "deploy/rk3588/validation/dpvo_m16_online" / path.name
    spec = importlib.util.spec_from_file_location("_test_v42_EOF_compiler", path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


@pytest.fixture(scope="module")
def parent():
    return Path(os.environ.get("V42_PARENT_COMPILER",
                               HERE.parent / "dpvo_grouped_update/compile-v37-grouped-update.py"))


class FakeToolkit:
    def __init__(self, codes=None):
        self.calls = []
        self.codes = codes or {}

    def record(self, name, args):
        self.calls.append((name, args))
        return self.codes.get(name, 0)

    def config(self, **args):
        return self.record("config", args)

    def load_onnx(self, **args):
        return self.record("load_onnx", args)

    def build(self, **args):
        return self.record("build", args)

    def export_rknn(self, path):
        return self.record("export_rknn", path)


def test_exact_original_nonquantized_operations(compiler, parent, tmp_path):
    tool, operations = FakeToolkit(), []
    model, output = tmp_path / "never-read.onnx", tmp_path / "never-written.rknn"
    compiler.compile_operations(tool, model, output, operations, parent_compiler=parent)
    assert tool.calls == [
        ("config", dict(target_platform="rk3588", float_dtype="float16", optimization_level=3)),
        ("load_onnx", dict(model=str(model))),
        ("build", dict(do_quantization=False)),
        ("export_rknn", str(output))]
    assert [row["name"] for row in operations] == [name for name, _ in tool.calls]
    assert all(row["code"] == 0 and row["seconds"] >= 0 for row in operations)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["config", "load_onnx", "build", "export_rknn"])
def test_failed_operation_stops_at_exact_boundary(compiler, parent, tmp_path, failure):
    tool, operations = FakeToolkit({failure: -1}), []
    with pytest.raises(RuntimeError, match=failure + " returned -1"):
        compiler.compile_operations(tool, tmp_path / "in", tmp_path / "out", operations,
                                    parent_compiler=parent)
    assert tool.calls[-1][0] == failure
    assert operations[-1]["name"] == failure and operations[-1]["code"] == -1
    assert len(tool.calls) == ["config", "load_onnx", "build", "export_rknn"].index(failure) + 1


def test_original_none_success_return_kept(compiler, parent, tmp_path):
    tool = FakeToolkit({name: None for name in ("config", "load_onnx", "build", "export_rknn")})
    operations = []
    compiler.compile_operations(tool, tmp_path / "in", tmp_path / "out", operations,
                                parent_compiler=parent)
    assert len(operations) == 4 and all(row["code"] is None for row in operations)


def test_parent_bytes_checked_before_import(compiler, tmp_path):
    path = tmp_path / "bad-compiler.py"
    path.write_text("raise RuntimeError('must not import')\n")
    with pytest.raises(ValueError, match="SHA differs"):
        compiler.compile_operations(FakeToolkit(), tmp_path / "in", tmp_path / "out", [],
                                    parent_compiler=path)


@pytest.mark.parametrize("key", ["B2_D3_M16", "B1_D1_M16", "B1_D4_M16"])
def test_only_actual_D3_shape_accepted_at_cli(compiler, tmp_path, key):
    with pytest.raises(SystemExit) as result:
        compiler.main(["--manifest", "unused", "--manifest-sha256", "unused",
                       "--parent-manifest", "unused", "--model-key", key,
                       "--contracts-sha256", "unused", "--output-dir", str(tmp_path / "not-created")])
    assert result.value.code == 2
    assert not (tmp_path / "not-created").exists()


def test_frozen_online_contracts_load_without_toolkit(compiler):
    directory = Path(os.environ["V42_ONLINE_CONTRACTS_DIR"])
    path = directory / "v42_online_contracts.py"
    value = compiler.load_contracts(path, hashlib.sha256(path.read_bytes()).hexdigest(),
        v41_contracts=directory / "v41_online_contracts.py",
        m16_contracts=directory / "v40_m16_grouped_contracts.py",
        parent_contracts=directory / "v37_grouped_contracts.py")
    assert value.NEW_KEYS == compiler.NEW_KEYS
    assert value.COMPILE_SCHEMA == "v42_online_M16_EOF_Update_RKNN_compilation_diagnostic"
    assert value.CONFIGURATION == dict(target_platform="rk3588", float_dtype="float16",
                                      optimization_level=3, do_quantization=False)
