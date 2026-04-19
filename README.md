# Simulación de Fútbol Robótico - Avance del Proyecto

Este proyecto consiste en una simulación de un equipo de fútbol compuesto por 11 jugadores robóticos. El objetivo principal de este avance es demostrar la coordinación multi-agente, donde cada jugador funciona como un proceso de Python independiente.

## ¿Qué se implementó en este avance?

En esta etapa, nos enfocamos en establecer la arquitectura base del sistema y la gestión de procesos:

*   **Arquitectura de Procesos:** Se logró que cada uno de los 11 jugadores se ejecute de forma autónoma, permitiendo que operen en paralelo sin interferir directamente con los demás.
*   **Definición de Roles:** Implementamos una estructura de clases para diferenciar el comportamiento según la posición:
    *   1 Portero (Goalkeeper)
    *   4 Defensas (Defenders)
    *   6 Atacantes (Attackers)
*   **Gestión Centralizada:** Se implemento el script `manager.py` encargado de lanzar y monitorear todos los procesos del equipo de manera eficiente.

## Arquitectura del Modelo (IA)

Cada jugador posee un "cerebro" basado en una red neuronal profunda (**Deep Q-Network - DQN**) que le permite tomar decisiones en tiempo real basándose en su visión del entorno.

### Estructura de la Red Neuronal
La arquitectura de la red neuronal es de tipo **Feedforward** y cuenta con las siguientes capas:
*   **Capa de Entrada:** Recibe 28 datos de visión (posiciones, distancias, etc.).
*   **Capa Oculta 1:** 128 neuronas con activación ReLU.
*   **Capa Oculta 2:** 128 neuronas con activación ReLU.
*   **Capa Oculta 3:** 64 neuronas con activación ReLU.
*   **Capa de Salida:** 14 neuronas que representan las posibles jugadas o acciones maestras del jugador.

### Proceso de Aprendizaje
El modelo utiliza **Aprendizaje por Refuerzo (Reinforcement Learning)**:
1.  **Percepción:** El jugador recibe el estado actual del campo (28 entradas).
2.  **Decisión:** El modelo elige una de las 14 acciones. Se utiliza una estrategia *Epsilon-Greedy* (10% de exploración aleatoria para probar nuevas jugadas).
3.  **Retroalimentación:** Tras realizar la acción, el jugador recibe una recompensa.
4.  **Ajuste:** Mediante el algoritmo de retropropagación (backpropagation) y optimización de errores, las neuronas ajustan sus pesos para mejorar futuras decisiones.

## Resultados Obtenidos

Como resultado, el sistema es capaz de inicializar el equipo completo en pocos segundos. Cada jugador identifica su rol y comienza a ejecutar su lógica de comportamiento, la cual se puede observar a través de los logs en la consola. 

Puedes ver una demostración del funcionamiento en el siguiente enlace:
**[Ver Video Demostrativo](https://youtu.be/hfiZ2SuE3QY)**

## Distribución del Proyecto

El código está organizado de manera sencilla para facilitar su mantenimiento:

*   `manager.py`: El punto de entrada que orquesta a todos los jugadores.
*   `players/brain.py`: Contiene la lógica del "cerebro" (DQN) y la arquitectura de la red neuronal.
*   `players/`: Carpeta que contiene la lógica individual (`base_player.py`, `defender.py`, `attacker.py`, `goalkeeper.py`).
*   `pesos_jugadores/`: Directorio donde se almacenan los pesos aprendidos (`.npz`) por cada rol.
*   `requirements.txt`: Lista de dependencias necesarias (NumPy es la principal para los cálculos de la IA).

## Instalación y Uso

### Configuración del Entorno
Se recomienda usar un entorno virtual:

```bash
python3 -m venv env
source env/bin/activate
```

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Ejecución
Para poner en marcha la simulación, simplemente ejecuta:
```bash
python3 manager.py
```

