import numpy as np
import os

#  DQN Deep Q-Network
# Esta clase es el sistema nervioso de nuestros jugadores. 
# Aquí es donde se procesan los 28 datos de visión para elegir una de las 14 jugadas.
class ManualBrain:
    def __init__(self, input_size=28, h1=128, h2=128, h3=64, output_size=14, learning_rate=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = learning_rate

        # INICIALIZACIÓN DE NEURONAS Pesos y Sesgos
        # Usamos una técnica llamada 'He' para que las neuronas no empiecen "dormidas".
        # W1: Capa de entrada (recibe los 28 datos)
        self.W1 = np.random.randn(input_size, h1) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, h1))
        # W2 y W3: Capas intermedias que analizan patrones (como "si hay un rival cerca, corre")
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2./h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, h3) * np.sqrt(2./h2)
        self.b3 = np.zeros((1, h3))
        # W4: Capa de salida (las 14 posibles jugadas maestras)
        self.W4 = np.random.randn(h3, output_size) * 0.01
        self.b4 = np.zeros((1, output_size))

        self.total_lessons = 0     # Contador de cuántas veces ha practicado
        self.reward_history = []   # Memoria de largo plazo para las gráficas
        self.temp_rewards = []     # Memoria de corto plazo (últimas 100 jugadas)

    #  Si una idea es negativa, la ignora (pone a 0).
    def relu(self, x):
        return np.maximum(0, x)

    #  Sirve para saber cuánto "empujar" una neurona durante el aprendizaje.
    def d_relu(self, x):
        return (x > 0).astype(float)

    # Convierte los números en porcentajes de probabilidad (0% a 100%).
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-8)

    # Aquí los datos viajan a través de las neuronas para generar una decisión.
    def forward(self, x):
        self.x = np.nan_to_num(np.array(x).reshape(1, -1))
        
        # Paso por la primera capa (128 neuronas)
        self.z1 = np.dot(self.x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Paso por la segunda capa (128 neuronas)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # Paso por la tercera capa (64 neuronas)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.relu(self.z3)
        
        # Generar las 14 opciones finales (Q-Values)
        self.q_values = np.dot(self.a3, self.W4) + self.b4
        return self.q_values[0]

    # A veces elige la mejor jugada, y a veces prueba algo nuevo (Epsilon) para aprender.
    def select_action(self, state, epsilon=0.1):
        q_values = self.forward(state)
        if np.random.rand() < epsilon:
            return np.random.randint(self.output_size) # Prueba algo loco
        return np.argmax(q_values) # Haz lo que crees que es mejor

    # Aquí el jugador compara lo que hizo con la recompensa recibida y ajusta sus neuronas.
    def update_rl(self, state, action_idx, reward):
        reward = np.clip(reward, -10, 10) # No queremos castigos ni premios exagerados
        self.temp_rewards.append(reward)
        
        current_q = self.forward(state)
        target_q = current_q.copy()
        target_q[action_idx] = reward # Ajustamos la meta
        
        # Calculamos el error: ¿Qué tan lejos estuvimos de la jugada perfecta?
        d_q = (current_q - target_q).reshape(1, -1)

        # RETROPROPAGACIÓN: Ajustamos cada capa desde el final hacia el principio.
        dW4 = np.dot(self.a3.T, d_q)
        db4 = d_q
        d_a3 = np.dot(d_q, self.W4.T)
        d_z3 = d_a3 * self.d_relu(self.z3)
        
        dW3 = np.dot(self.a2.T, d_z3)
        db3 = d_z3
        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * self.d_relu(self.z2)
        
        dW2 = np.dot(self.a1.T, d_z2)
        db2 = d_z2
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.d_relu(self.z1)
        
        dW1 = np.dot(self.x.T, d_z1)
        db1 = d_z1

        # CLIPPING: Evitamos que el cerebro explote si el error es muy grande.
        for grad in [dW1, db1, dW2, db2, dW3, db3, dW4, db4]:
            np.clip(grad, -1, 1, out=grad)

        #  Las neuronas "aprenden" un poquito con el Learning Rate.
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
        self.W4 -= self.lr * dW4; self.b4 -= self.lr * db4
        
        self.total_lessons += 1
        # Guardamos el promedio de humor del jugador cada 100 lecciones
        if self.total_lessons % 100 == 0:
            self.reward_history.append(np.mean(self.temp_rewards))
            self.temp_rewards = []

    # Para que el equipo no olvide lo aprendido
    def save_weights(self, role):
        filename = f"pesos_jugadores/{role}/shared_weights.npz"
        if not os.path.exists(os.path.dirname(filename)): os.makedirs(os.path.dirname(filename))
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, 
                 W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4, 
                 lessons=self.total_lessons, history=np.array(self.reward_history))

    def load_weights(self, role):
        filename = f"pesos_jugadores/{role}/shared_weights.npz"
        if os.path.exists(filename):
            data = np.load(filename)
            try:
                self.W1, self.b1 = data['W1'], data['b1']
                self.W2, self.b2 = data['W2'], data['b2']
                self.W3, self.b3 = data['W3'], data['b3']
                self.W4, self.b4 = data['W4'], data['b4']
                self.total_lessons = int(data['lessons'])
                self.reward_history = list(data['history'])
                print(f"[{role}] Cerebro DQN cargado con {self.total_lessons} lecciones.")
            except: print(f"[{role}] Error cargando pesos.")
