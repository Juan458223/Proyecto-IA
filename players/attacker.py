from base_player import BasePlayer
import sys
import numpy as np
import time

class Attacker(BasePlayer):
    #Sistema de Recompensas
    def calculate_reward(self):
        reward = -0.02 # Costo por perder el tiempo
        
        # ganar puntos por acercarse al balón
        if self.ball_dist < 20:
            reward += (20 - self.ball_dist) * 0.1
            
        #  Patear cerca del arco rival da muchos puntos
        if self.kickable:
            reward += 10.0
            if self.goal_rival_dist < 20:
                reward += 50.0 # Bono por oportunidad de gol
        
        # Mirar a portería 
        if self.ball_dist < 2 and abs(self.goal_rival_angle) < 30:
            reward += 2.0
            
        return reward

    def action_loop(self):
        print(f"[{self.role}] Olfateando el gol con cerebro DQN...")
        while True:
            msg = self.receive(timeout=0.01)
            if msg: self.parse_message(msg)
            
            # Si el árbitro dice que esperemos, volvemos a nuestra posición
            if "before_kick_off" in self.play_mode or "goal_" in self.play_mode:
                x, y = self.start_position()
                self.send(f"(move {x} {y})")
            else:
                # 1. Mirar el mundo (28 datos)
                state = self.get_state_vector()
                # 2. El cerebro elige una de las 14 jugadas
                action_idx = self.brain.select_action(state)
                # 3. Mover los músculos
                self.execute_action(action_idx)
                
                time.sleep(0.05) # Pequeña pausa para ver el resultado
                # 4. Aprender
                new_reward = self.calculate_reward()
                self.brain.update_rl(state, action_idx, new_reward)
                
                # Guardar el conocimiento cada 50 jugadas
                if self.brain.total_lessons % 50 == 0:
                    self.brain.save_weights(self.role)

            time.sleep(0.05)

if __name__ == "__main__":
    player = Attacker(team_name="MYTEAMUD", number=int(sys.argv[1]), role=sys.argv[2])
    player.start()
