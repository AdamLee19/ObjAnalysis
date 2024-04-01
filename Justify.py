# import Mesh, MeshIO



import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from NumpyIO import writeFile, readFile
import cv2 as cv

import argparse
import time
import concurrent.futures
from math import ceil



def main(start: int, end: int) -> None:

    print(start, end, end='\n\n\n')
    dataFolder = Path('./combine')
    

    for i in range(start, end):
        total_points = 12466 
        faceName = Path(f'Face{i:0>{5}}')
        dir = dataFolder / faceName
        

        leftDir = dir / Path('left_visible.npy')
        frontDir = dir / Path('front_visible.npy')
        rightDir = dir / Path('right_visible.npy')
        

        leftIdx = readFile(leftDir)
        frontIdx = readFile(frontDir)
        rightIdx = readFile(rightDir)
        
        idx = np.zeros(total_points, dtype=int)
        idx[leftIdx] = 1
        writeFile(dir / Path('left_visi.npy'), idx)

        idx = np.zeros(total_points, dtype=int)
        idx[frontIdx] = 1
        writeFile(dir / Path('front_visi.npy'), idx)

        idx = np.zeros(total_points, dtype=int)
        idx[rightIdx] = 1
        writeFile(dir / Path('right_visi.npy'), idx)


        

       

if __name__ == "__main__":
    from threading import Thread
    

    parser = argparse.ArgumentParser(description='Visibility')
    parser.add_argument('--start', type=int, required=True, help="Start face")
    parser.add_argument('--end',  type=int, required=True, help="End face")
    args = parser.parse_args()
    
    #main('left', 0, 2)
    import os
    num_threads =os.cpu_count()
    print(num_threads)

    start = args.start
    end = args.end

    interval = ceil((end - start) / num_threads)
    
    pairs = []
    for i in range(num_threads):
        s = i * interval + start
        e = s + interval if (s + interval) <= end else end 
        
        pairs.append((s, e))    

    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(main, s, e) for s, e in pairs]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    



       

       

    

    

    

    
    
    
    
    
    
   
   
    