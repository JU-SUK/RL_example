import gym
from sac_learn import SACagent

def main():
    max_episode_num = 200
    env = gym.make('Pendulum-v1')
    agent = SACagent(env)

    agent.train(max_episode_num)

    agent.plot_result()

if __name__ == "__main__":
    main()