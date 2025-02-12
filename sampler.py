import numpy as np
import pandas as pd

class Sampler:
    def __init__(self, sol, user_df, poi_density, transition_to_index):
        self.sol = sol
        self.user_df = user_df
        self.poi_density = poi_density
        self.transition_to_index = transition_to_index

    def match(self):
        home, work = self._pick_home_work()
        if self.user_df.loc[0]['act_type'] == 1:
            locations = [home]
        elif self.user_df.loc[0]['act_type'] == 2:
            locations = [work]
        else:
            locations = [self._init_sample(self.user_df.loc[0]['act_type'])]
        
        prev_loc = locations[0]

        for idx, row in self.user_df.iterrows():
            if pd.isna(row['next_activity']):
                break
            if row['next_activity'] == 1:
                next_loc = home

            elif row['next_activity'] == 2:
                next_loc = work

            else:
                next_loc = self._sample(row['act_type'], row['next_activity'], prev_loc)
            
            locations.append(next_loc)
            prev_loc = next_loc
        
        self.user_df['binary_subdivision_id_divided_by_5'] = locations
        return self.user_df

    def parse_result(self):
        res = []
        for user, locs in self.all_matched.items():
            for loc in locs:
                res.append({'agent_id': user, 'location': loc})
        self.res = pd.DataFrame(res)
        return self.res

    def _init_sample(self, act_type):
        den = self.poi_density[act_type]
        return np.random.choice(np.arange(len(den)), p=den)
        
    
    def _pick_home_work(self):
        home = np.random.choice(np.arange(len(self.poi_density[1])), p=self.poi_density[1])
        work_prob = self.sol[home,:,self.transition_to_index[(1, 2)]]
        work_prob = abs(work_prob+0.0000001) / abs(work_prob+0.0000001).sum()
        work = np.random.choice(np.arange(len(work_prob)), p=work_prob)
        return home, work

    def _sample(self, prev_act, next_act, origin):
        transition_index = self.transition_to_index[(prev_act, next_act)]
        prob = self.sol[origin, :, transition_index]
        prob = abs(prob+0.0000001) / abs(prob+0.0000001).sum()
        destination = np.random.choice(np.arange(len(prob)), p=prob)
        
        return destination