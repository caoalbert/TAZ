import sys
import os
import pickle
import multiprocessing
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sampler import Sampler

# ps aux | grep parallel_spill.py
# nohup python run_sampler_non_par.py > output.log &
# ls -1 | wc -l
# output/demand_variation/spill_op_result/alpha_7_demand_500_maxwait_10 
# killall -u albert

def batch_match(users, batch_id, sol, df, poi_density, transition_to_index):
    print(f"Batch {batch_id} started")
    start_time = time.time()
    results = pd.DataFrame()
    for user in users:
        test = Sampler(sol, df[df['agent_id'] == user].reset_index(drop=True), poi_density, transition_to_index)
        test_df = test.match()
        results = pd.concat([results, test_df])
    end_time = time.time()
    print(f"Batch {batch_id} finished in {end_time - start_time} seconds")
    
    results.to_csv(f"output/matched_{batch_id}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", "-n", type=int, default=24)
    args = parser.parse_args()
    num_processes = args.n_cores  # Number of CPU cores

    # Load data
    df = pd.read_parquet('data/activity.parquet')
    df['next_activity'] = df.groupby('agent_id')['act_type'].shift(-1)
    df_grouped = df.groupby(['act_type', 'next_activity']).count().reset_index()[['act_type','next_activity','agent_id']]
    df_grouped = df_grouped.rename(columns={'agent_id':'count'})

    with open('input/poi_density.pkl', 'rb') as f:
        poi_density = pickle.load(f)

    sol = np.load('input/sol_top_100.npy')

    transition_to_index = {}
    for idx, row in df_grouped.iterrows():
        transition_to_index[(row['act_type'], row['next_activity'])] = idx

    users = df['agent_id'].unique()
    batch_size = 10000

    input_to_process = []
    for i in range(0, len(users), batch_size):
        input_to_process.append((users[i:i+batch_size], i//batch_size, sol, df, poi_density, transition_to_index))

    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(batch_match, input_to_process)
