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

import logging 
import sys

class ShortNameFilter(logging.Filter):
    def filter(self, record):
        record.shortname = record.name.replace("radpattern.", "")
        return True

def setup_run_logging(output_dir):
    """Handles logging to file and console"""

    # Creates log info file 
    output_dir = Path(output_dir)

    log_path = output_dir / "run.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicated logs if setup_run_logging() is called twice
    root.handlers.clear()


    fmt = "%(asctime)s | %(levelname)s | %(shortname)s | %(message)s"
    datefmt = "%H:%M:%S"

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # File: INFO, WARNING, ERROR...
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ShortNameFilter())

    # Console: only WARNING, ERROR...
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ShortNameFilter())

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging initialized. Log file: %s", log_path)

    return log_path

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

            setp = objs.sim.sim_metadataSetUp(objs.exp, objs.Cbeam)
            mc_folder = output_dir / f"{setp.run_name}_mc_runs"

            setup_run_logging(mc_folder)
            log = logging.getLogger(__name__)

            log.info("Starting config: %s", config_path)
            log.info("Output folder: %s", mc_folder)
            
            mc_folder = run_one_config(objs, output_dir, save_full_mc = True)
            log.info("Finished config: %s", config_path)

            print("done:", config_path)
            try: 
                dst = mc_folder / config_path.name
                shutil.copy2(config_path, dst)
            except (TypeError, FileNotFoundError) as e:
                print(f" Couldn't copy json file to mc_runs folder. Error {e}")


            move_file(config_path, done_dir)


        except Exception:
            print("failed:", config_path)

            failed_path = move_file(config_path, failed_dir)

            error_path = failed_path.with_suffix(".error.txt")
            error_path.write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
