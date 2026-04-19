from base_player import BasePlayer
import sys
import numpy as np
import time

# --- EL GUARDIÁN ---
class Goalkeeper(BasePlayer):
    # ¿CÓMO SE SIENTE EL PORTERO? (Sistema de Recompensas)
    def calculate_reward(self):
        reward = -0.05 # Costo por tiempo
        
        # Premiar estar muy cerca de su propio arco (Su zona de confort)
        if self.goal_own_dist < 5:
            reward += 5.0
        elif self.goal_own_dist > 25:
            reward -= 5.0 # Penalizar si se va a pasear lejos del arco
            
        # Recompensa gigante por atajar el balón cerca del arco
        if self.kickable and self.goal_own_dist < 15:
            reward += 100.0
            
        # Premiar que siempre esté mirando al balón (Atención)
        if abs(self.ball_angle) < 30:
            reward += 1.0
            
        return reward

    # EL CICLO DE JUEGO
    def action_loop(self):
        print(f"[{self.role}] Protegiendo la red con cerebro DQN...")
        while True:
            msg = self.receive(timeout=0.01)
            if msg: self.parse_message(msg)
            
            # Al inicio o tras un gol, vuelve al centro de su arco
            if "before_kick_off" in self.play_mode or "goal_" in self.play_mode:
                x, y = self.start_position()
                self.send(f"(move {x} {y})")
            else:
                state = self.get_state_vector()
                # El cerebro DQN decide cómo atajar
                action_idx = self.brain.select_action(state)
                # Ejecutar acción (Turn para mirar balón, Dash para tapar, etc.)
                self.execute_action(action_idx)
                
                time.sleep(0.05)
                new_reward = self.calculate_reward()
                self.brain.update_rl(state, action_idx, new_reward)
                
                if self.brain.total_lessons % 50 == 0:
                    self.brain.save_weights(self.role)

            time.sleep(0.05)

if __name__ == "__main__":
    player = Goalkeeper(team_name="MYTEAMUD", number=int(sys.argv[1]), role=sys.argv[2])
    player.start()
