import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import random
import numpy as np
import numpy as np

class Dataset(object):
    def __init__(self, data_map, deterministic=False, shuffle=True):
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle: self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, deterministic=True):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, deterministic)


def iterbatches(arrays, *, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle: np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')

print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.v_pred = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.v_pred[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 8),
                            nn.ReLU(),
                            nn.Linear(8, 8),
                            nn.ReLU(),
                            nn.Linear(8, action_dim),
                            nn.ReLU()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 8),
                            nn.ReLU(),
                            nn.Linear(8, 8),
                            nn.ReLU(),
                            nn.Linear(8, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        vpred = self.critic(state)

        return action.detach(), action_logprob.detach(), vpred.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        vpred = self.critic(state)
        
        return action_logprobs,dist_entropy, vpred 


class PPO:
    def __init__(self, state_dim, 
                    action_dim,
                    lr_actor, 
                    lr_critic, 
                    gamma, 
                    K_epochs, 
                    eps_clip, 
                    has_continuous_action_space, 
                    action_std,
                    LAMBDA,
                    entropy_coeff,
                    batch_size):

        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std
        self.lmda = LAMBDA
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim,has_continuous_action_space, action_std).to(device)
        self.critic_opt = torch.optim.Adam([
                       
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic, 'eps' : 1e-5}
                    ])
        self.policy_opt = torch.optim.Adam([ {'params': self.policy.actor.parameters(), 'lr': lr_actor, 'eps' : 1e-5}])

        self.policy_old = ActorCritic(state_dim, action_dim,has_continuous_action_space, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten() , state_val.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)


            return action.item(), state_val.detach().cpu().numpy().flatten()

    def update(self, **kwargs):
        truncation_size = len(self.buffer.states)-1
        last_state_v_pred = kwargs.get("next_v_pred")

        # Monte Carlo estimate of returns
        seg = {"ob": torch.tensor([self.buffer.states[0].cpu().clone().tolist()  for _ in range(truncation_size)]),
               "ac": torch.tensor([self.buffer.actions[0].cpu().clone().tolist()  for _ in range(truncation_size)]),
                "old_logprobs": torch.tensor([self.buffer.logprobs[0].cpu().clone()  for _ in range(truncation_size)]),
               "rew": torch.zeros(truncation_size, dtype=float),
               "v_pred": torch.zeros(truncation_size, dtype=float),
               "done": torch.zeros(truncation_size, dtype=int),
               }
        
        for t in range(truncation_size):
            seg["ob"][t] = self.buffer.states[t].cpu().clone()
            seg["ac"][t] = self.buffer.actions[t].cpu().clone()
            seg["old_logprobs"][t] = self.buffer.logprobs[t].cpu().clone()
            seg["rew"][t] = self.buffer.rewards[t]
            seg["done"][t] = self.buffer.is_terminals[t]
            seg["v_pred"][t] = self.buffer.state_values[t]


        seg_done = seg["done"]
        vpred = np.append(seg["v_pred"], last_state_v_pred) # currently we add 0
        gae_lam = torch.empty(truncation_size, dtype = float)
        seg_rewards = seg["rew"]
        last_gae_lam = 0
        for t in reversed(range(truncation_size)):
            non_terminal = 1 - seg_done[t]
            delta = seg_rewards[t] + self.gamma * vpred[t + 1] * non_terminal - vpred[t]
            gae_lam[t] = delta + self.gamma * self.lmda * non_terminal * last_gae_lam
            last_gae_lam = gae_lam[t]

        seg["adv"] = gae_lam
        seg["td_lam_ret"] = seg["adv"] + seg["v_pred"]
        self.learn(ob=seg["ob"], 
                   ac=seg["ac"], 
                   adv=seg["adv"],
                   old_logprobs= seg["old_logprobs"],
                   td_lam_ret=seg["td_lam_ret"],
                   )
        
    def learn(self, **kwargs):
        bs = kwargs.get("ob")
        ba = kwargs.get("ac")
        batch_adv = kwargs.get("adv")
        batch_td_lam_ret = kwargs.get("td_lam_ret")
        batch_adv = (batch_adv - batch_adv.mean()) / batch_adv.std()
        old_logprobs = kwargs.get("old_logprobs")

        d = Dataset(dict(ob=bs, 
                         ac=ba, 
                         atarg=batch_adv, 
                         vtarg=batch_td_lam_ret,
                         old_logprobs = old_logprobs),

                         deterministic=False)
        
        batch_size = self.batch_size or bs.shape[0]
        self.policy_old.load_state_dict(self.policy.state_dict())

        for _ in range(self.K_epochs):
            for batch in d.iterate_once(batch_size):
                atarg = batch["atarg"]
                action_logprobs, dist_entropy, vpred = self.policy.evaluate(batch["ob"], batch["ac"])

                mean_ent = torch.mean(dist_entropy)

                ratio = torch.exp(action_logprobs - batch["old_logprobs"])
                surr1 = torch.where(
                    torch.logical_or(torch.isinf(ratio * atarg ), torch.isnan(ratio * atarg)),
                    torch.zeros_like(ratio),
                    ratio * atarg
                    )
                           
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * atarg
                # final loss of clipped objective PPO
                vf_loss = torch.mean(torch.square(vpred - batch["vtarg"])) 
                pol_surr = torch.mean(torch.minimum(surr1, surr2))
                pol_ent_pen = -1 * self.entropy_coeff * mean_ent
                total_loss = pol_surr + pol_ent_pen + vf_loss

                self.critic_opt.zero_grad()
                self.policy_opt.zero_grad()

                vf_loss.backward(retain_graph=True)
                total_loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.pi.vf_shaped.parameters(), 1.0)
                # torch.nn.utils.clip_grad_norm_(self.pi.pol.parameters(), 1.0)

                self.critic_opt.step()
                self.policy_opt.step()
            
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


