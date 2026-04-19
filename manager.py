import subprocess
import time
import threading

# --- EL ENTRENADOR (Manager) ---
# Este script es el encargado de llamar a todos los jugadores, 
# asignarles su número y asegurarse de que si alguno se cae, vuelva a levantarse.

# Definimos la alineación inicial: 1 Portero, 4 Defensas, 6 Atacantes (Total 11)
ROLES = {"goalkeeper": 1, "defender": 4, "attacker": 6}

def launch_player(role, number):
    """
    Lanza el proceso de un jugador usando el Python del entorno virtual (.venv).
    Esto asegura que todas las librerías como Numpy estén disponibles.
    """
    path = f"players/{role}.py"
    venv_python = "./.venv/bin/python3" 
    
    # Abrimos un túnel de comunicación con el proceso del jugador
    proc = subprocess.Popen(
        [venv_python, path, str(number), role],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    print(f"  -> {role.capitalize()} #{number} saltando al campo (pid={proc.pid})")
    return {"proc": proc, "number": number, "role": role}

def monitor_process(player):
    """
    Escucha lo que dice cada jugador y lo imprime en nuestra consola.
    """
    proc = player["proc"]
    while True:
        line = proc.stdout.readline()
        if line:
            print(f"[{player['role']}-{player['number']}] {line.strip()}")
        err = proc.stderr.readline()
        if err:
            print(f"[{player['role']}-{player['number']}] ERR: {err.strip()}")
        if proc.poll() is not None: # Si el proceso terminó
            break

def main():
    processes = []
    player_number = 1

    print("=== INICIANDO EQUIPO PROFESIONAL DQN ===")

    # 1. LANZAR AL PORTERO (Dorsal 1)
    processes.append(launch_player("goalkeeper", player_number))
    player_number += 1

    # 2. LANZAR A LOS DEFENSAS (Dorsales 2 al 5)
    for _ in range(ROLES["defender"]):
        processes.append(launch_player("defender", player_number))
        player_number += 1

    # 3. LANZAR A LOS ATACANTES (Dorsales 6 al 11)
    for _ in range(ROLES["attacker"]):
        processes.append(launch_player("attacker", player_number))
        player_number += 1

    print("\n¡Todo el equipo está en la cancha! Monitoreando rendimiento...\n")

    # Creamos hilos para que los mensajes de todos los jugadores se vean al mismo tiempo
    for player in processes:
        threading.Thread(target=monitor_process, args=(player,), daemon=True).start()

    # BUCLE DE SUPERVISIÓN: Si alguien se cae (crash), lo reiniciamos de inmediato.
    try:
        while True:
            time.sleep(1)
            for i, player in enumerate(processes[:]):
                proc = player["proc"]
                if proc.poll() is not None:
                    print(f"❌ [{player['role']}-{player['number']}] se ha retirado. ¡Reiniciando!")
                    new_player = launch_player(player["role"], player["number"])
                    processes[i] = new_player
                    threading.Thread(target=monitor_process, args=(new_player,), daemon=True).start()
    except KeyboardInterrupt:
        print("\nEntrenador cerrando sesión. Retirando al equipo...")
        for player in processes:
            player["proc"].terminate()
        print("Hecho.")

if __name__ == "__main__":
    main()
