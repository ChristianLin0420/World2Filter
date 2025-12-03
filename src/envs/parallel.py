import multiprocessing as mp
import numpy as np
import gym

class ParallelEnv:
    """
    Parallel environment wrapper using multiprocessing.
    
    Runs multiple environments in separate processes for parallel data collection.
    """
    def __init__(self, env_fns, start_method='fork'):
        """
        Args:
            env_fns: List of callables that create environments
            start_method: Multiprocessing start method ('fork' or 'spawn')
        """
        self.num_envs = len(env_fns)
        
        # Create dummy env to get specs
        dummy_env = env_fns[0]()
        
        # Support both embodied (obs_space/act_space) and gym (observation_space/action_space) naming
        self.observation_space = getattr(dummy_env, 'observation_space', None) or getattr(dummy_env, 'obs_space', None)
        self.action_space = getattr(dummy_env, 'action_space', None) or getattr(dummy_env, 'act_space', None)
        self.obs_space = getattr(dummy_env, 'obs_space', None) or self.observation_space
        self.act_space = getattr(dummy_env, 'act_space', None) or self.action_space
        
        # Determine action dim
        if hasattr(dummy_env, 'action_dim'):
            self.action_dim = dummy_env.action_dim
        elif isinstance(self.action_space, dict) and 'action' in self.action_space:
            # Embodied format
            self.action_dim = self.action_space['action'].shape[0]
        elif isinstance(self.action_space, dict) and 'shape' in self.action_space:
            self.action_dim = self.action_space['shape'][0]
        elif hasattr(self.action_space, 'shape'):
            self.action_dim = self.action_space.shape[0]
        elif hasattr(self.action_space, 'n'):
             self.action_dim = self.action_space.n
        else:
            dummy_env.close()
            raise ValueError("Could not determine action dimension")
            
        dummy_env.close()
            
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [
            ctx.Process(target=self._worker, args=(work_remote, remote, env_fn))
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def _worker(self, remote, parent_remote, env_fn):
        parent_remote.close()
        env = env_fn()
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    next_obs, reward, terminated, truncated, info = env.step(data)
                    if terminated or truncated:
                        # Auto-reset for continuous training
                        # But return the terminal observation in info for proper handling
                        # Note: we usually return the FIRST observation of the NEW episode as 'next_obs'
                        # and store the terminal observation in info['final_observation']
                        final_obs = next_obs
                        next_obs, reset_info = env.reset()
                        info['final_observation'] = final_obs
                        info['reset_info'] = reset_info
                    remote.send((next_obs, reward, terminated, truncated, info))
                elif cmd == 'reset':
                    obs, info = env.reset()
                    remote.send((obs, info))
                elif cmd == 'close':
                    env.close()
                    remote.close()
                    break
                elif cmd == 'getattr':
                    remote.send(getattr(env, data))
                else:
                    raise NotImplementedError(f"Unknown command: {cmd}")
        except Exception as e:
            print(f"Worker failed: {e}")
            remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminated, truncated, infos = zip(*results)
        
        # Stack results
        obs = np.stack(obs)
        rewards = np.stack(rewards)
        terminated = np.stack(terminated)
        truncated = np.stack(truncated)
        # infos is a tuple of dicts
        
        return obs, rewards, terminated, truncated, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def __getattr__(self, name):
        self.remotes[0].send(('getattr', name))
        return self.remotes[0].recv()
