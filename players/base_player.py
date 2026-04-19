import socket
import time
import re
import math
import numpy as np
from brain import ManualBrain

# Configuración del simulador para la conexion
SERVER_HOST = "localhost"
SERVER_PORT = 6000
BUFFER_SIZE = 8192

# Esta clase maneja cómo el jugador ve el mundo y cómo mueve sus músculos.
class BasePlayer:
    def __init__(self, team_name, number, role):
        self.team_name = team_name
        self.number = number
        self.role = role
        self.side = "l"
        self.server_port = SERVER_PORT

        # Inicializamos el cerebro profesional (28 datos de entrada, 14 jugadas posibles)
        self.brain = ManualBrain(input_size=28, output_size=14)
        self.brain.load_weights(self.role)

        # ESTADO FÍSICO
        self.pos_x, self.pos_y = 0.0, 0.0      # Mi ubicación en el mapa
        self.vel_x, self.vel_y = 0.0, 0.0      # Mi velocidad actual
        self.body_angle = 0.0                  # Hacia dónde apunta mi pecho
        self.stamina = 8000.0                  # Mi energía (si llega a 0, no puedo correr)
        self.kickable = 0                      # ¿Tengo el balón a mis pies?

        # RADAR DE OBJETOS
        self.ball_dist, self.ball_angle = 100.0, 0.0
        self.ball_vel_x, self.ball_vel_y = 0.0, 0.0
        
        # Listas para guardar a los 3 rivales y compañeros más cercanos
        self.rivals = [(100.0, 0.0)] * 3
        self.teammates = [(100.0, 0.0)] * 3
        
        # Ubicación de los arcos
        self.goal_rival_dist, self.goal_rival_angle = 100.0, 0.0
        self.goal_own_dist, self.goal_own_angle = 100.0, 0.0

        self.play_mode = "before_kick_off"     # Estado del partido (esperando, jugando, etc.)
        self.game_time = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Hablar con el servidor del simulador.
    def send(self, msg):
        self.sock.sendto(msg.encode(), (SERVER_HOST, self.server_port))

    #  Escuchar lo que el árbitro y mis ojos ven.
    def receive(self, timeout=None):
        self.sock.settimeout(timeout)
        try:
            data, addr = self.sock.recvfrom(BUFFER_SIZE)
            if addr[1] != self.server_port: self.server_port = addr[1]
            return data.decode()
        except: return ""

    # Traducir los códigos del servidor a lenguaje humano.
    def parse_message(self, msg):
        if "(init " in msg: self._parse_init(msg)
        elif "(sense_body " in msg: self._parse_sense_body(msg)
        elif "(see " in msg: self._parse_see(msg)
        elif "(hear " in msg: self._parse_hear(msg)

    def _parse_init(self, msg):
        parts = re.findall(r"[\w\_]+", msg)
        if len(parts) >= 4:
            self.side = parts[1]
            self.play_mode = parts[3]

    def _parse_sense_body(self, msg):
        m = re.search(r"\(stamina ([\d\.]+)", msg)
        if m: self.stamina = float(m.group(1))

    def _parse_hear(self, msg):
        m_ref = re.search(r"referee\s+([\w\_]+)", msg)
        if m_ref: self.play_mode = m_ref.group(1)

    def _parse_see(self, msg):
        # Limpiamos el radar antes de cada parpadeo
        self.rivals = [(100.0, 0.0)] * 3
        self.teammates = [(100.0, 0.0)] * 3
        
        # Usamos los arcos para saber dónde estamos parados.
        m_goal_l = re.search(r"\(\(g l\)\s+([\d\.-]+)\s+([\d\.-]+)", msg)
        m_goal_r = re.search(r"\(\(g r\)\s+([\d\.-]+)\s+([\d\.-]+)", msg)
        
        if m_goal_l:
            d, a = float(m_goal_l.group(1)), float(m_goal_l.group(2))
            if self.side == "l": 
                self.goal_own_dist, self.goal_own_angle = d, a
                self.pos_x = -52.5 + d * math.cos(math.radians(a)) # Estoy cerca de mi meta izquierda
                self.pos_y = d * math.sin(math.radians(a))
            else: 
                self.goal_rival_dist, self.goal_rival_angle = d, a
                self.pos_x = 52.5 - d * math.cos(math.radians(a))
                self.pos_y = -d * math.sin(math.radians(a))

        if m_goal_r:
            d, a = float(m_goal_r.group(1)), float(m_goal_r.group(2))
            if self.side == "r": 
                self.goal_own_dist, self.goal_own_angle = d, a
                self.pos_x = 52.5 - d * math.cos(math.radians(a))
                self.pos_y = -d * math.sin(math.radians(a))
            else: 
                self.goal_rival_dist, self.goal_rival_angle = d, a
                self.pos_x = -52.5 + d * math.cos(math.radians(a))
                self.pos_y = d * math.sin(math.radians(a))

        # VER EL BALÓN
        m_ball = re.search(r"\(\(b\)\s+([\d\.-]+)\s+([\d\.-]+)", msg)
        if m_ball:
            self.ball_dist = float(m_ball.group(1))
            self.ball_angle = float(m_ball.group(2))
            self.kickable = 1 if self.ball_dist < 1.0 else 0

        # Radar de amigos y enemigos
        players = re.findall(r"\(p\s+\"([\w\_]+)\"\s+([\d\.-]+)\s+([\d\.-]+)", msg)
        rivals_found = []
        teammates_found = []
        for p in players:
            team, dist, ang = p[0], float(p[1]), float(p[2])
            if team == self.team_name: teammates_found.append((dist, ang))
            else: rivals_found.append((dist, ang))
        
        # Ordenamos por cercanía para saber quién nos va a quitar el balón
        rivals_found.sort()
        for i in range(min(3, len(rivals_found))): self.rivals[i] = rivals_found[i]
        teammates_found.sort()
        for i in range(min(3, len(teammates_found))): self.teammates[i] = teammates_found[i]

    # Creamos la lista de 28 números normalizados (entre -1 y 1) para que el cerebro los entienda.
    def get_state_vector(self):
        state = [
            self.pos_x/52.5, self.pos_y/34.0,       # Ubicación propia
            self.vel_x/1.5, self.vel_y/1.5,         # Mi velocidad
            self.body_angle/180.0,                  # Hacia dónde miro
            self.stamina/8000.0,                    # Energía restante
            float(self.kickable),                   # ¿Tengo el balón?
            self.ball_dist/100.0, self.ball_angle/180.0, # Ubicación del balón
            self.ball_vel_x/3.0, self.ball_vel_y/3.0,     # Velocidad del balón
            1.0 if self.pos_x < 0 else 0.0,         # ¿El balón está en peligro?
        ]
        # Agregamos los datos de los 3 rivales y 3 compañeros
        for r in self.rivals: state.extend([r[0]/100.0, r[1]/180.0])
        for t in self.teammates: state.extend([t[0]/100.0, t[1]/180.0])
        # Ubicación de los arcos
        state.extend([self.goal_rival_dist/100.0, self.goal_rival_angle/180.0])
        state.extend([self.goal_own_dist/100.0, self.goal_own_angle/180.0])
        return state

    # Traduce el número elegido por el cerebro (0-13) en un comando real.
    def execute_action(self, action_idx):
        if action_idx == 0: self.send("(dash 100)")             # Sprint máximo
        elif action_idx == 1: self.send("(dash 50)")            # Trote medio
        elif action_idx == 2: self.send("(dash 20)")            # Caminar
        elif action_idx == 3: self.send("(turn 90)")            # Giro brusco derecha
        elif action_idx == 4: self.send("(turn 45)")            # Giro leve derecha
        elif action_idx == 5: self.send("(turn -45)")           # Giro leve izquierda
        elif action_idx == 6: self.send("(turn -90)")           # Giro brusco izquierda
        elif action_idx == 7: self.send(f"(kick 100 {self.goal_rival_angle})") # ¡TIRO AL ARCO!
        elif action_idx == 8: self.send(f"(kick 60 {self.teammates[0][1]})")   # Pase al amigo
        elif action_idx == 9: self.send("(kick 100 0)")          # Despeje de emergencia
        elif action_idx == 10: self.send("(turn 180)")          # Dar media vuelta
        elif action_idx == 11: self.send("(turn 10)")           # Ajuste fino mirada
        elif action_idx == 12: self.send("(turn -10)")          # Ajuste fino mirada
        elif action_idx == 13: self.send("(dash 0)")            # Esperar/Analizar

    # POSICIONES DE LA FORMACIÓN 
    def start_position(self):
        positions = {
            1: (-50, 0),    2: (-35, -15), 3: (-35, 0), 4: (-35, 15), 5: (-35, 20),
            6: (-20, -20),  7: (-20, 20),  8: (-15, -10), 9: (-15, 10), 10: (-5, -5), 11: (-5, 5)
        }
        return positions.get(self.number, (0, 0))

    # ACTIVAR EL JUGADOR
    def start(self):
        init_msg = f"(init {self.team_name} (goalie))" if self.role == "goalkeeper" else f"(init {self.team_name})"
        self.send(init_msg)
        self.action_loop()
