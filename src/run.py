import random
import numpy as np
import torch

import exercise_1 as e1
import exercise_2 as e2
import exercise_3 as e3
import exercise_4 as e4
import exercise_5 as e5
import exercise_6 as e6
import exercise_7 as e7
import exercise_8 as e8

# Set random seed for reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv"

def main():
    print("Starting the League of Legends Match Predictor...")
    print("Starting exercise 1")
    e1.run_exercise_1(DATASET_URL)
    print("Starting exercise 2")
    e2.run_exercise_2()
    print("Starting exercise 3")
    e3.run_exercise_3()
    #print("Starting exercise 4")
    #e4.run_exercise_4()
    print("Starting exercise 5")
    e5.run_exercise_5()
   # print("Starting exercise 6")
   # e6.run_exercise_6()
   # print("Starting exercise 7")
   # e7.run_exercise_7()
   # print("Starting exercise 8")
   # e8.run_exercise_8()
    print("League of Legends Match Predictor completed successfully.")
main()










