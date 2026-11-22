"""d3d12_xs_surface: the 116-slot export table and the 83 standard GUIDs.

    python -m pytest tests/test_d3d12_xs_surface.py -q
    python -I tests/test_d3d12_xs_surface.py

The SCARAB cross-check runs only when SCARAB_REF points at a checkout; it is not needed for CI.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import d3d12_xs_surface as S  # noqa: E402


def test_slots_are_116():
    assert S.check_slots() == []


def test_data_classes():
    assert S.check_data_classes() == []


def test_guid_values():
    assert S.check_guid_values() == []


def test_iid_block_is_alphabetical():
    assert S.check_iid_order() == []


def test_standard_iids_match_sdk():
    um, _shared = S.find_sdk_include()
    if not um:
        raise unittest.SkipTest("Windows SDK headers not found")
    errors, checked, _um = S.check_sdk()
    assert errors == []
    assert checked == 83


def test_sdk_include_override_accepts_a_trailing_separator():
    # WINDOWS_SDK_INCLUDE may name the version folder or its um folder, with or without a
    # trailing separator; all four spellings must find the same um/shared pair.
    with tempfile.TemporaryDirectory() as tmp:
        version = os.path.join(tmp, "10.0.99999.0")
        um = os.path.join(version, "um")
        os.makedirs(um)
        os.makedirs(os.path.join(version, "shared"))
        saved = os.environ.get("WINDOWS_SDK_INCLUDE")
        try:
            for value in (um, um + os.sep, version, version + os.sep):
                os.environ["WINDOWS_SDK_INCLUDE"] = value
                got_um, got_shared = S.find_sdk_include()
                assert got_um is not None, value
                assert os.path.normcase(got_um) == os.path.normcase(um), (value, got_um)
                assert os.path.normcase(got_shared) == os.path.normcase(
                    os.path.join(version, "shared")), (value, got_shared)
        finally:
            if saved is None:
                os.environ.pop("WINDOWS_SDK_INCLUDE", None)
            else:
                os.environ["WINDOWS_SDK_INCLUDE"] = saved


def test_scarab_cross_check():
    root = os.environ.get("SCARAB_REF")
    if not root:
        raise unittest.SkipTest("SCARAB_REF is not set")
    errors, _notes = S.check_scarab(root)
    assert errors == []


if __name__ == "__main__":
    test_slots_are_116()
    test_data_classes()
    test_guid_values()
    test_iid_block_is_alphabetical()
    test_sdk_include_override_accepts_a_trailing_separator()
    um, _shared = S.find_sdk_include()
    if um:
        errors, checked, _um = S.check_sdk()
        assert errors == [], errors
        assert checked == 83
    if os.environ.get("SCARAB_REF"):
        errors, _notes = S.check_scarab(os.environ["SCARAB_REF"])
        assert errors == [], errors
    else:
        print("SCARAB cross-check skipped: SCARAB_REF is not set")
    print("ok")
