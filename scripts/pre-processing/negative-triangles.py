""" 
For each gml file in the source folder, generate a json file in the 
target folder containing a sample of negative triangles for use in triangle 
detection tasks.
"""
import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

from gtl import Graph

PROJECT_DIR: Path = Path(__file__).parent.parent.parent.resolve()

# pyre-ignore[9]:
target_dir: Path = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("target")

    opts = parser.parse_args()

    src_dir = Path(opts.src)

    global target_dir
    target_dir = Path(opts.target)

    if not src_dir.is_dir():
        print("The src path must exist and be a directory")
        sys.exit(1)

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    paths = src_dir.glob("*.gml")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_sample_from_gml, paths)

        for result in results:
            pass


def _sample_from_gml(path: Path) -> None:
    filename: str = path.stem
    destination: Path = Path(target_dir) / f"{filename}-negative-triangles.json"
    print(f"{destination}")

    g = Graph.from_gml_file(path)
    sample = g.sample_negative_triangles(len(g.get_triangles_list()))
    with open(destination, "w") as f:
        json.dump(sample, f)


if __name__ == "__main__":
    main()
