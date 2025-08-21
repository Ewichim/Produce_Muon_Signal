'''
Author: Anthony Badea, Eliza Howard
Date: 01/08/25
'''

import subprocess
import os
import argparse
import sys

def run_commands(commands):
    for command in commands:
        print("Running:", command)
        # detect pixelAV commands by checking for the pixelAV directory in the first element
        if isinstance(command, list) and os.path.isdir(command[0]):
            # pixelAV: run from its directory
            subprocess.run(command[1:], cwd=command[0])
        else:
            subprocess.run(command)

if __name__ == "__main__":

    # user options
    parser = argparse.ArgumentParser(usage=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--outDir", help="Output directory",
                        default="./Simulation_Output_Beam_Spot")
    parser.add_argument("-p", "--pixelAVdir",
                        help="pixelAV directory",
                        default="~/PixelSim2/pixelav/")
    ops = parser.parse_args()

    # get absolute path for pixelAV directory
    pixelAVdir = os.path.expanduser(ops.pixelAVdir)

    # get absolute path and check if outdir exists
    outDir = os.path.abspath(ops.outDir)
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    else:
        response = input("Folder exists\nTo empty folder and create new files, enter \"yes\": ")
        if response != "yes":
            print("\n\nExiting program...\n\n")
            sys.exit()
        # Empty folder
        for f in os.listdir(outDir):
            path = os.path.join(outDir, f)
            if os.path.isfile(path):
                os.remove(path)

    commands = []

    tracklist_folder = os.path.abspath("./Tracklists")

    # build command sequences
    for sub in os.listdir(tracklist_folder):
        folder = os.path.abspath(os.path.join(tracklist_folder, sub))
        tag0 = "bib" if "BIB" in folder else "sig"
        i = 0

        for tracklist in os.listdir(folder):
            tracklist_path = os.path.join(folder, tracklist)
            tag = f"{tag0}{i}"
            outFileName = os.path.join(outDir, tag)

            # pixelAV command
            pixelAV_cmd = [
                pixelAVdir,
                "./bin/ppixelav2_list_trkpy_n_2f.exe",
                "1",
                tracklist_path,
                f"{outFileName}.out",
                f"{outFileName}_seed"
            ]

            # parquet-writing command
            parquet_cmd = [
                "python3",
                "./processing/datagen.py",
                "-f",
                f"{tag}.out",
                "-t",
                tag,
                "-d",
                outDir
            ]

            # store as a two-step sequence
            commands.append([pixelAV_cmd, parquet_cmd])
            i += 1

    # run all commands sequentially on a single thread
    for cmd_seq in commands:
        run_commands(cmd_seq)
