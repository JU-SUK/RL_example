import gym
from ddpg_learn import DDPGagent

def main():
    max_episode_num = 200
    env = gym.make("Pendulum-v1")
    agent = DDPGagent(env)

    # 학습 진행
    agent.train(max_episode_num)

    # 결과 도시
    agent.plot_result()

if __name__ == "__main__":
    main()