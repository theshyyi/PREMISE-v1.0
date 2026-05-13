from pathlib import Path

from premise.binaryio import parse_meta_file


def test_parse_meta_file(tmp_path: Path):
    meta = tmp_path / "sample.meta"
    meta.write_text("var=pr\nunits=mm/day\nnx=2\nny=2\n", encoding="utf-8")
    result = parse_meta_file(str(meta))
    assert result["var"] == "pr"
    assert result["units"] == "mm/day"
    assert result["nx"] == "2"
