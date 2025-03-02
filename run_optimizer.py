import numpy as np
import pandas as pd
import pickle
from model import *
import argparse

# pkill -f "run_optimizer.py"
# nohup python run_optimizer.py > run_optimizer.out &

# import os
# os.chdir('TAZ/')


activity_seq = {
    'HOME': 1, 'WORK': 2, 'SCHOOL': 3, 'CHILDCARE': 4, 'BUYGOODS': 5, 'SERVICES': 6,
    'EATOUT': 7, 'ERRANDS': 8, 'RECREATION': 9, 'EXERCISE': 10, 'VISIT': 11, 'HEALTHCARE': 12,
    'RELIGIOUS': 13, 'SOMETHINGELSE': 14, 'DROPOFF': 15, 'TRANSPORTATION': 16, 'PORTAL': 17
}
def mapper(x):
    return activity_seq[x]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ActChainPath", "-a", type=str, default="data/activity_postprocessed.parquet")
    parser.add_argument("--OutPath", "-o", type=str, default="input/sol_0301")
    args = parser.parse_args()

    with open('input/poi_density.pkl', 'rb') as f:
        poi_density = pickle.load(f)
    df = pd.read_parquet(args.ActChainPath)    
    df['act_type'] = df['act_type'].apply(mapper)
    df['hour_of_day'] = df['start_timestep'].apply(map_to_hour_of_day)
    df = df[df['act_type'] != 17].reset_index(drop=True)
    df['act_type'] = df['act_type'].cat.remove_unused_categories() 
    df['next_activity'] = df.groupby('agent_id')['act_type'].shift(-1)

    for p in ['MD', 'PM', 'RD']:
        print(f'{p} started')
        od = pd.read_parquet('data/veraset_tokyo_trips_binary_subdivision_id.parquet')
        od['trip_start_datetime'] = pd.to_datetime(od['trip_start_datetime'])
        od['period'] = od['trip_start_datetime'].dt.hour.apply(assign_time_period)
        od = od[od['period'] == p]
        od['start_binary_subdivision_id'] = od['start_binary_subdivision_id'] // 2 // 2 // 2 // 2 // 2
        od['end_binary_subdivision_id'] = od['end_binary_subdivision_id'] // 2 // 2 // 2 // 2 // 2
        od = od.groupby(['start_binary_subdivision_id', 'end_binary_subdivision_id']).size().reset_index()
        od.columns = ['origin', 'destination', 'flow']

        df_p = df[df['hour_of_day'] == p]
        df_grouped = df_p.groupby(['act_type', 'next_activity']).count().reset_index()[['act_type','next_activity','agent_id']]
        df_grouped = df_grouped.rename(columns={'agent_id':'count'})
        total_num_transitions = df_grouped['count'].sum()
        df_grouped_sorted = df_grouped.sort_values('count', ascending=False).head(50).reset_index()

        od_matrix, in_flow_concentration, out_flow_concentration = compute_params(df_grouped, poi_density, od)
        sol = optimize(od['destination'].nunique(), df_grouped.shape[0], od_matrix, df_grouped, df_grouped_sorted, in_flow_concentration, out_flow_concentration)
        np.save(f'{args.OutPath}/{p}.npy', sol)
        print(f'{p} done')
