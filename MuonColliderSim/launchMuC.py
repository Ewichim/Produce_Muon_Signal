'''
Author: Anthony Badea, Eliza Howard
Date: 01/08/25
'''

import subprocess
import multiprocessing
import numpy as np
import os
import argparse
import sys

def run_executable(executable_path, options):
    command = [executable_path] + options
    subprocess.run(command)

def run_commands(commands):
    for command in commands:
        print(command)
        if "pixelav" in command[0]:
            subprocess.run(command[1:], cwd=command[0])
        else:
            subprocess.run(command)
    

if __name__ == "__main__":

    # user options
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--outDir", help="Output directory", default="./Simulation_Output")
    parser.add_argument("-j", "--ncpu", help="Number of cores to use", default=20, type=int)
    parser.add_argument("-p", "--pixelAVdir", help="pixelAV directory", default="/home/youeric/PixelSim2/pixelav")
    ops = parser.parse_args()

    # get absolute path for semiparametric directory
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
        files = os.listdir(outDir)
        for f in files:
            if os.path.isfile(f"{outDir}/{f}"):
                os.remove(f"{outDir}/{f}")

    commands = []

    tracklist_folder=os.path.abspath("./Tracklists")

    for folder in os.listdir(tracklist_folder):

        folder = os.path.abspath(os.path.join(tracklist_folder, folder))

        if "BIB" in folder:
            tag0 = "bib"
        else:
            tag0 = "sig"
        
        i = 0

        for tracklist in os.listdir(folder):

            tracklist = os.path.abspath(os.path.join(folder, tracklist))

            #if tag0 == 'sig' and i > 76:
            #    break
            
            tag = f"{tag0}{i}"

            outFileName = f"{outDir}/{tag}"

            # Run pixelAV
            pixelAV = [pixelAVdir, "./bin/ppixelav2_list_trkpy_n_2f_custom.exe", "1", tracklist, f"{outFileName}.out", f"{outFileName}_seed"]

            # Write parquet file
            parquet = ["python3", "./processing/datagen.py", "-f", f"{tag}.out", "-t", tag, "-d", outDir]

            # commands
            commands.append([(pixelAV, parquet,),]) # weird formatting is because pool expects a tuple at input

            i += 1       
             
        
    # List of CPU cores to use for parallel execution
    num_cores = multiprocessing.cpu_count() if ops.ncpu == -1 else ops.ncpu

    # Create a pool of processes to run in parallel
    pool = multiprocessing.Pool(num_cores)
    
    # Launch the executable N times in parallel with different options
    # pool.starmap(run_executable, [(path_to_executable, options) for options in options_list])
    print(commands) # Anthony you are here need to make the multiprocess work with delphes tied in
    pool.starmap(run_commands, commands)
    
    # Close the pool of processes
    pool.close()
    pool.join()
