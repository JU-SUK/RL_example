import gym
import tensorflow as tf
from a3c_learn import A3Cagent

def main():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    agent = A3Cagent(env_name)
    # 글로벌 신경망 파라미터 가져옴
    agent.load_weights('./save_weights/')
    time = 0
    state = env.reset()[0]

    while True:
        env.render()
        # 행동 계산
        action = agent.global_actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        # 환경으로부터 다음 상태, 보상 받음
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break
    env.close()

if __name__ == "__main__":
    main()