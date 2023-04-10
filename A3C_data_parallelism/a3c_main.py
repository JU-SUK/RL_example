from a3c_learn import A3Cagent

def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v1'
    agent = A3Cagent(env_name)
    # 학습 진행
    agent.train(max_episode_num)
    # 학습 결과 도시
    agent.plot_result()

if __name__ == "__main__":
    main()