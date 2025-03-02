import sys
import os
import pickle
import multiprocessing
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sampler import Sampler, sample_home
from model import *

# ps aux | grep parallel_spill.py
# nohup python -u run_sampler.py > output.out 2>&1 &
# ls -1 | wc -l
# output/demand_variation/spill_op_result/alpha_7_demand_500_maxwait_10 
# killall -u albert

activity_seq = {
    'HOME': 1, 'WORK': 2, 'SCHOOL': 3, 'CHILDCARE': 4, 'BUYGOODS': 5, 'SERVICES': 6,
    'EATOUT': 7, 'ERRANDS': 8, 'RECREATION': 9, 'EXERCISE': 10, 'VISIT': 11, 'HEALTHCARE': 12,
    'RELIGIOUS': 13, 'SOMETHINGELSE': 14, 'DROPOFF': 15, 'TRANSPORTATION': 16, 'PORTAL': 17
}
def mapper(x):
    return activity_seq[x]

def batch_match(households, batch_id, sol_matrices, df, poi_density, transition_to_index, output_path):
    print(f"Batch {batch_id} started", flush=True)
    start_time = time.time()
    results = pd.DataFrame()

    for household in households:
        home = sample_home(poi_density)
        users = df[df['household_id'] == household]['agent_id'].unique()
        for user in users:
            test = Sampler(sol_matrices, df[df['agent_id'] == user].reset_index(drop=True), poi_density, transition_to_index)
            test_df = test.match(home)
            results = pd.concat([results, test_df])

    end_time = time.time()
    print(f"Batch {batch_id} finished in {np.round(end_time - start_time,2)} seconds", flush=True)
    
    results.to_csv(f"{output_path}/matched_{batch_id}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", "-n", type=int, default=24)
    parser.add_argument("--ActChainPath", "-a", type=str, default="data/activity_postprocessed.parquet")
    parser.add_argument("--MatPath", "-m", type=str, default="input/sol_0301")
    parser.add_argument("--OutputPath", "-o", type=str, default="output/household_0301")
    args = parser.parse_args()


    # Load data
    df = pd.read_parquet(args.ActChainPath)
    df['act_type'] = df['act_type'].apply(mapper)
    df['next_activity'] = df.groupby('agent_id')['act_type'].shift(-1)
    df['hour_of_day'] = df['start_timestep'].apply(map_to_hour_of_day)

    # Filter out transition from home to home
    # df = df[~((df['act_type'] == 1) & (df['next_activity'] == 1))].reset_index(drop=True)

    df_grouped = df[df['act_type'] != 17]
    df_grouped['act_type'] = df_grouped['act_type'].cat.remove_unused_categories()
    df_grouped = df_grouped.groupby(['act_type', 'next_activity']).count().reset_index()[['act_type','next_activity','agent_id']]
    df_grouped = df_grouped.rename(columns={'agent_id':'count'})

    with open('input/poi_density.pkl', 'rb') as f:
        poi_density = pickle.load(f)

    sol_matrices = []
    for p in range(["AM","MD","PM","RD"]):
        sol_matrices.append(np.load(f'{args.MatPath}/{p}.npy'))

    transition_to_index = {}
    for idx, row in df_grouped.iterrows():
        transition_to_index[(row['act_type'], row['next_activity'])] = idx

    households = df['household_id'].unique()

    batch_size = 5000
    input_to_process = []
    for i in range(0, len(households), batch_size):
        input_to_process.append((households[i:i+batch_size], i//batch_size, sol_matrices, df, poi_density, transition_to_index, args.OutputPath))
    
    with multiprocessing.Pool(args.n_cores) as pool:
        pool.starmap(batch_match, input_to_process)
