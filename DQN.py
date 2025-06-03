import numpy as np
import random
import time
import matplotlib.pyplot as plt

# import torch
import tensorflow as tf

from collections import deque

from lunar import LunarLanderEnv

# Lecturas interesantes: 
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (Playing atari with DQN)
# https://www.nature.com/articles/nature14236 (Human level control through RL)
# https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained

class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.oculta1 = tf.keras.layers.Dense(hidden_size, activation="relu", input_shape=(state_size,)) # explicado en https://keras.io/2/api/layers/core_layers/dense/
        self.oculta2 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.salida = tf.keras.layers.Dense(action_size, activation="linear")
    
    def call(self, inputs): # Función call requerida según https://keras.io/api/models/model/
        res = self.oculta1(inputs)
        res = self.oculta2(res)
        return self.salida(res)
    
class ReplayBuffer():
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size) # deque es una doble cola que permite añadir y quitar elementos de ambos extremos

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states = np.vstack(states).astype(np.float32)
        next_states = np.vstack(next_states).astype(np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent():
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.3,
                learning_rate=0.001, batch_size=64, 
                memory_size=10000, episodes=3000, 
                target_network_update_freq=10,
                replays_per_episode=1000):
        """
        Initialize the DQN agent with the given parameters.
        
        Parameters:
        lunar (LunarLanderEnv): The Lunar Lander environment instance.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for exploration rate.
        epsilon_min (float): Minimum exploration rate.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of the batch for experience replay.
        memory_size (int): Number of experiences stored on the replay memory.
        episodes (int): Number of episodes to train the agent.
        target_network_update_freq (int): Frequency of updating the target network.
        """
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 1/episodes
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        
        self.target_updt_freq = target_network_update_freq
        self.replays_per_episode = replays_per_episode
        
        # Initialize replay memory
        # a deque is a double sided queue that allows us to append and pop elements from both ends
        self.memory = ReplayBuffer(memory_size)
        
        # Initialize the environment
        self.lunar = lunar
        
        observation_space = lunar.env.observation_space
        action_space = lunar.env.action_space
        
        # La red neuronal debe tener un numero de parametros
        # de entrada igual al espacio de observaciones
        # y un numero de salida igual al espacio de acciones.
        # Asi como un numero de capas intermedias adecuadas.
        hidden_size = 64
        
        self.q_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=hidden_size #elegir un tamaño de capa oculta
        )
        
        self.target_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=hidden_size #elegir un tamaño de capa oculta
        )
        
        # Set weights of target network to be the same as those of the q network
        self.target_network.set_weights(self.q_network.get_weights())
      
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        
        print(f"QNetwork:\n {self.q_network}")
          
    def act(self):
        """
        This function takes an action based on the current state of the environment.
        it can be randomly sampled from the action space (based on epsilon) or
        it can be the action with the highest Q-value from the model.
        """
        state_input = np.expand_dims(self.lunar.state, axis=0).astype(np.float32)

        if np.random.rand() <= self.epsilon:
            action = self.lunar.env.action_space.sample()  # Exploración
        else:
            q_values = self.q_network(state_input)
            action = np.argmax(q_values[0].numpy())  # Explotación

        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        self.memory.push(self.lunar.state, action, reward, next_state, done)
        return next_state, reward, done, action
    
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
        """
        
       # Si no hay suficientes muestras, no entrenar
        if len(self.memory) < self.batch_size:
            return None

        # 1. Obtener batch aleatorio del buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 2. Convertir a tensores
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)

        # 3. Calcular el target Q-value: Q_target = r + γ * max(Q(next_state))
        next_q_values = self.target_network(next_states_tensor)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        q_targets = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)

        with tf.GradientTape() as tape:
            # 4. Calcular los Q-values actuales para las acciones tomadas
            q_values = self.q_network(states_tensor)
            indices = tf.stack([tf.range(self.batch_size), actions_tensor], axis=1)
            selected_q_values = tf.gather_nd(q_values, indices)

            # 5. Calcular la pérdida (loss)
            loss = tf.keras.losses.MSE(q_targets, selected_q_values)

        # 6. Backpropagation y actualización de pesos
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        return loss.numpy()
        
    def update_target_network(self):
        # copiar los pesos de la red q a la red objetivo
         self.target_network.set_weights(self.q_network.get_weights())
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
        self.q_network.save_weights(path)
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        # cargar el modelo desde el path indicado
        dummy_input = tf.convert_to_tensor(np.zeros((1, self.lunar.env.observation_space.shape[0])), dtype=tf.float32)
        self.q_network(dummy_input)
        self.target_network(dummy_input)

        # Cargar los pesos
        self.q_network.load_weights(path)
        self.target_network.load_weights(path)
        
    def train(self):
        """
        Train the DQN agent on the given environment for a specified number of episodes.
        The agent will interact with the environment, store experiences in memory, and learn from them.
        The target network will be updated periodically based on the update freq parameter.
        The agent will also decay the exploration rate (epsilon) over time.
        The training process MUST be logged to the console.    
        Returns:
        None
        """
        
        rewards_per_episode_tot = np.zeros(self.episodes)
        rewards_per_episode_pos = np.zeros(self.episodes)
        sum_rewards_tot = np.zeros(self.episodes)
        sum_rewards_pos = np.zeros(self.episodes)
        epsilon_history = []

        for episode in range(self.episodes):
            state = self.lunar.reset()
            total_reward = 0

            for step in range(self.replays_per_episode):
                # Elegir acción y realizarla
                next_state, reward, done, action = self.act()
                next_state, reward, done = self.lunar.take_action(action, verbose=False)

                # Almacenar en buffer
                self.memory.push(state, action, reward, next_state, done)

                # Entrenamiento
                if len(self.memory) >= self.batch_size:
                    self.update_model()

                state = next_state
                total_reward += reward

                if done:
                    break

            # Decaimiento de epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon - self.epsilon_decay
                epsilon_history.append(self.epsilon)

            # Actualizar red objetivo periódicamente
            if episode % self.target_updt_freq == 0:
                self.update_target_network()

            print(f"Episode {episode + 1}/{self.episodes} - Total Reward: {total_reward:.2f} - Epsilon: {self.epsilon:.4f}")

            rewards_per_episode_tot[episode] = total_reward
            rewards_per_episode_pos[episode] = max(0, total_reward)
            sum_rewards_tot[episode] = np.sum(rewards_per_episode_tot[max(0, episode-100):(episode+1)])
            sum_rewards_pos[episode] = np.sum(rewards_per_episode_pos[max(0, episode-100):(episode+1)])
            
            if(episode % 100 == 0):
                self.save_model(f"modelol3000ep-64hid{episode}.weights.h5")
                print(f"Modelo episodio {episode} guardado en modelol3000ep-64hid{episode}.weights.h5 con recompensa {sum_rewards_tot[episode]:.2f}")

        plt.figure(1)

        plt.subplot(131)
        plt.plot(sum_rewards_tot)
        plt.subplot(132)
        plt.plot(sum_rewards_pos)
        plt.subplot(133)
        plt.plot(epsilon_history)
        plt.savefig('ejemplo.png')

        # Guardar el modelo al terminar
        self.save_model("modelol3000ep-64hidfinal.weights.h5")
        print("Modelo guardado en 'modelol3000ep-64hidfinal.weights.h5'")