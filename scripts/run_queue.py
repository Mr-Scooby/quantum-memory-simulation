#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Look into a queue folder and picks the json files, constructs the objects for the simulation 
and calls the runner to run the simulation based on the file. 

creates folder in destination with the sim_name and stores the mc_runs.npz, parent mc run npz file
and the json file. 

loops over the queue folder until no more files in it. 
copies json file to done/ directory for tracking. 
if error arises moves file to failed/ directory with a txt error traceback and continues to next file 

"""


from pathlib import Path
import shutil
import traceback

from radpattern.config.builder import build_run_objects
from radpattern.simulation.runner import run_one_config 

def move_file(src, dst_dir):
    """
    Move one file into a folder.

    Output
    ------
    Path
        New file path.
    """

    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name

    if dst.exists():
        dst = dst_dir / f"{src.stem}_copy{src.suffix}"

    shutil.move(str(src), str(dst))

    return dst


def main():
    queue_dir = Path(r"C:\Users\local_admin\radek\simulations\tests\locals_runs\queue")
    done_dir = Path(r"C:\Users\local_admin\radek\simulations\tests\locals_runs\done")
    failed_dir = Path(r"C:\Users\local_admin\radek\simulations\tests\locals_runs\failed")
    output_dir = Path(r"C:\Users\local_admin\radek\simulations\data\test")

    print(queue_dir.glob)
    for config_path in sorted(queue_dir.glob("*.json")):
        print("running:", config_path)

        try:

            objs = build_run_objects(str(config_path))
            
            mc_folder = run_one_config(objs, output_dir, save_full_mc = True)

        except Exception:
            print("failed:", config_path)

            failed_path = move_file(config_path, failed_dir)

            error_path = failed_path.with_suffix(".error.txt")
            error_path.write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )

        else:
            print("done:", config_path)
            try: 
                dst = mc_folder / config_path.name
                shutil.copy2(config_path, dst)
            except (TypeError, FileNotFoundError) as e:
                print(f" Couldn't copy json file to mc_runs folder. Error {e}")


            move_file(config_path, done_dir)


if __name__ == "__main__":
    main()
