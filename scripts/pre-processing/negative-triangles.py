""" 
For each gml file in the source folder, generate a json file in the 
target folder containing a sample of negative triangles for use in triangle 
detection tasks.

This requires alot (>32GB) of RAM!!
"""
import json
from gtl import Graph
import sys
from pathlib import Path
import argparse

PROJECT_DIR: Path = Path(__file__).parent.parent.parent.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("target")

    opts = parser.parse_args()

    src_dir = Path(opts.src)
    target_dir = Path(opts.target)

    if not src_dir.is_dir():
        print("The src path must exist and be a directory")
        sys.exit(1)

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    # do this in single thread only due to high memory usage
    for path in src_dir.glob("*.gml"):
        print(path)
        filename: str = path.stem
        destination: Path = Path(f"{filename}-negative-triangles.json")

        g = Graph.from_gml_file(path)
        sample = g.sample_negative_triangles(len(g.get_triangles_list()))
        with open(destination, "w") as f:
            json.dump(sample, f)


if __name__ == "__main__":
    main()
