from ppo_learn import PPOagent
import gym

def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = PPOagent(env)

    # 학습 진행
    agent.train(max_episode_num)

    agent.plot_result()

if __name__ == "__main__":
    main()