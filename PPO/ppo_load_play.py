import gym
import tensorflow as tf
from ppo_learn import PPOagent

def main():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    agent = PPOagent(env)

    agent.load_weights('./save_weights/')

    time = 0
    state = env.reset()[0]

    while True:
        env.render()

        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break
    env.close()

if __name__ == "__main__":
    main()
