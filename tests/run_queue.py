#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import traceback

from config_object import build_run_objects

from jsonSim_parallelizationGpu_MC_TimeEvolution import run_one_config


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
                dst = output_dir / config_path.name
                shutil.copy2(config_path, dst)
            except (TypeError, FileNotFoundError) as e:
                print(f" Couldn't copy json file to mc_runs folder. Error {e}")


            move_file(config_path, done_dir)


if __name__ == "__main__":
    main()
