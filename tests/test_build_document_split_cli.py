from pathlib import Path

from document_splits.build_document_split import build_parser, resolve_paths


def test_root_dir_sets_scratch_layout_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--root-dir", str(tmp_path)])

    dataset, output = resolve_paths(args)

    assert dataset == tmp_path / "processed"
    assert output == tmp_path / "data/splits/document_split.json"


def test_explicit_paths_override_root_dir(tmp_path: Path) -> None:
    dataset_override = tmp_path / "custom-input"
    output_override = tmp_path / "custom-output.json"
    args = build_parser().parse_args(
        [
            "--root-dir",
            str(tmp_path / "root"),
            "--dataset",
            str(dataset_override),
            "--output",
            str(output_override),
        ]
    )

    dataset, output = resolve_paths(args)

    assert dataset == dataset_override
    assert output == output_override
