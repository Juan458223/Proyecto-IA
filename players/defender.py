from base_player import BasePlayer
import sys
import numpy as np
import time

class Defender(BasePlayer):
    # Sistema de Recompensas
    def calculate_reward(self):
        reward = -0.05 # Costo por tiempo
        
        # Premiamos estar bien parados (entre el balón y nuestro arco)
        if self.ball_dist < 30:
            angulo_cobertura = abs(self.ball_angle - self.goal_own_angle)
            if angulo_cobertura > 140:
                reward += 2.0 # Está tapando bien
            else:
                reward -= 0.5
                
        # Premiamos el marcaje personal a los rivales
        dist_rival = self.rivals[0][0]
        if dist_rival < 5:
            reward += 1.5 # Incomodando al delantero
            
        # Recompensa  por tener el balón
        if self.kickable:
            reward += 10.0
            if self.pos_x < 0:
                reward += 20.0 # Bono extra por despejar en zona de peligro
                
        # Mantener la disciplina táctica en su zona
        if -45 < self.pos_x < -15:
            reward += 1.0
        else:
            reward -= 0.5

        return reward

    def action_loop(self):
        print(f"[{self.role}] Protegiendo la casa con cerebro DQN...")
        while True:
            msg = self.receive(timeout=0.01)
            if msg: self.parse_message(msg)
            
            if "before_kick_off" in self.play_mode or "goal_" in self.play_mode:
                x, y = self.start_position()
                self.send(f"(move {x} {y})")
            else:
                state = self.get_state_vector()
                # El cerebro DQN elige la mejor jugada defensiva
                action_idx = self.brain.select_action(state)
                # Ejecutar la jugada (Dash, Turn, Kick, etc.)
                self.execute_action(action_idx)
                
                time.sleep(0.05)
                new_reward = self.calculate_reward()
                self.brain.update_rl(state, action_idx, new_reward)
                
                if self.brain.total_lessons % 50 == 0:
                    self.brain.save_weights(self.role)

            time.sleep(0.05)

if __name__ == "__main__":
    player = Defender(team_name="MYTEAMUD", number=int(sys.argv[1]), role=sys.argv[2])
    player.start()
