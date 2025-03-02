from gurobipy import Model, GRB
from gurobipy import quicksum
import gurobipy as gp
from tqdm import tqdm
import numpy as np


def compute_params(df_grouped, poi_density, od):
    out_flow_concentration = {}
    in_flow_concentration = {}
    for idx, row in df_grouped.iterrows():
        prev_act, nex_act = row['act_type'], int(row['next_activity'])
        out_flow_concentration[idx] = row['count'] * poi_density[prev_act]
        in_flow_concentration[idx] = row['count'] * poi_density[nex_act]
        
    N = od['destination'].nunique()
    L = df_grouped.shape[0]

    total_num_transitions = df_grouped['count'].sum()
    inflow = od.groupby('destination')['flow'].sum().reset_index()
    expansion_factor = total_num_transitions / inflow['flow'].sum()

    od_matrix = np.zeros((N, N))
    for idx, row in od.iterrows():
        od_matrix[int(row['origin']), int(row['destination'])] = row['flow'] * expansion_factor

    return od_matrix, in_flow_concentration, out_flow_concentration



def optimize(N, L, od_matrix, df_grouped, df_grouped_sorted, in_flow_concentration, out_flow_concentration):
    m = Model()
    x_ijl = [(i, j, l) for i in range(N) for j in range(N) for l in range(L)]
    x_ijl = m.addVars(x_ijl, vtype=GRB.CONTINUOUS, name='x_ijl')

    z_ij = [(i, j) for i in range(N) for j in range(N)]
    z_ij = m.addVars(z_ij, vtype=GRB.CONTINUOUS, name='z_ij')
    for i in range(N):
        for j in range(N):
            m.addConstr(z_ij[i, j] >= quicksum(x_ijl[i, j, l] for l in range(L))-od_matrix[i,j])
            m.addConstr(z_ij[i, j] >= -quicksum(x_ijl[i, j, l] for l in range(L))+od_matrix[i,j])

    for l in range(L):
        m.addConstr(quicksum(x_ijl[i, j, l] for j in range(N) for i in range(N)) == df_grouped['count'].iloc[l])

    for i in range(N):
        for transition in range(len(df_grouped_sorted)):
            transition_index = df_grouped_sorted['index'].iloc[transition]
            m.addConstr(quicksum(x_ijl[i,j,transition_index] for j in range(N)) == out_flow_concentration[transition_index][i])
            m.addConstr(quicksum(x_ijl[j,i,transition_index] for j in range(N)) == in_flow_concentration[transition_index][i])
        
    m.setObjective(quicksum(z_ij[i, j] for i in range(N) for j in range(N)), GRB.MINIMIZE)
    # m.setParam('OutputFlag', 0)
    m.optimize()

    x_ijl_sol = m.getAttr('x', x_ijl)
    sol = np.zeros((N, N, L))
    for i in range(N):
        for j in range(N):
            for l in range(L):
                sol[i, j, l] = x_ijl_sol[i, j, l]

    return sol

def assign_time_period(x):
    if (x >=7) & (x < 11):
        return 'AM'
    elif (x >= 11) & (x < 17):
        return 'MD'
    elif (x >= 17) & (x < 23):
        return 'PM'
    else:
        return 'RD'

def map_to_hour_of_day(x):
    x = (x / 4) % 24
    if (x >=7) & (x < 11):
        return 'AM'
    elif (x >= 11) & (x < 17):
        return 'MD'
    elif (x >= 17) & (x < 23):
        return 'PM'
    else:
        return 'RD'
    
