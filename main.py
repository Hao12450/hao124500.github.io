import pygame
import numpy as np
import pygame.gfxdraw
from numpy.linalg import pinv, svd
import math

class RoboticArm:
    def __init__(self):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3-DOF Robotic Arm Simulation with SVD Projection")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)  # For first SVD direction (U[:, 0])
        self.GREEN = (0, 255, 0)  # For second SVD direction (U[:, 1])
        self.GRAY = (150, 150, 150)
        self.ORANGE = (255, 165, 0)  # For projected target
        self.MAGENTA = (255, 0, 255)  # For required end effector location
        self.CYAN = (0, 255, 255)  # For current end effector location
        self.BUTTON_COLOR = (0, 200, 0)
        self.BUTTON_HOVER_COLOR = (0, 150, 0)
        self.BUTTON_TEXT_COLOR = (255, 255, 255)
        self.SLIDER_COLOR = (200, 0, 0)
        self.SLIDER_KNOB_COLOR = (0, 0, 0)
        
        # Robot parameters
        self.link_lengths = [150, 100, 80, 80]  # L1, L2, L3, L4
        self.angles = np.array([0.0, 0.0, 0.0, 0.0])  # θ1, θ2, θ3, θ4
        self.end_theta = np.sum(self.angles)
        self.target_theta = 0.1
        self.target = np.array([200.0, 0.0, 0.0])
        self.projected_target = None  # Store projected target when needed
        self.last_projection = None  # Store the last projection calculation
        
        #Singular condition number threshold
        self.gamma = 0.1 #needs tuning

        # SVD parameters
        self.SINGULARITY_THRESHOLD = 20  # Threshold for considering a direction singular
        self.projection_mode = True  # Whether to use projection mode
        
        # UI parameters
        self.offset_x = 400
        self.offset_y = self.height // 2
        self.selected_joint = None
        self.show_jacobian = True
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.latex_font = pygame.font.Font(None, 20)
        
        self.is_animating = False
        self.clock = pygame.time.Clock()

    def forward_kinematics(self, angles=None):
        if angles is None:
            angles = self.angles
            
        cum_angle = 0
        x, y = 0, 0
        positions = [(x, y)]
        
        for i, (length, angle) in enumerate(zip(self.link_lengths, angles)):
            cum_angle += angle
            x += length * np.cos(cum_angle)
            y += length * np.sin(cum_angle)
            positions.append((x, y))
        
        return positions

    def calculate_jacobian(self):
        positions = self.forward_kinematics()
        J = np.zeros((2, 4))
        
        for i in range(4):
            x_sum = sum(self.link_lengths[j] * np.cos(sum(self.angles[:j+1])) 
                       for j in range(i, 4))
            y_sum = sum(self.link_lengths[j] * np.sin(sum(self.angles[:j+1])) 
                       for j in range(i, 4))
            
            J[0, i] = -y_sum
            J[1, i] = x_sum
        
        r = np.array([[1, 1, 1, 1]])
        J = np.vstack((J, r))
        return J

    def project_target_on_valid_direction(self, current_pos, target, U, S):
        distance_to_target = np.linalg.norm(target)
        max_reach = sum(self.link_lengths)
        error_vector = target - current_pos
        sigma_max = max(S) #find the maximum singular value
        S_normalized = [sig/ sigma_max for sig in S] #normalize the singular values with respect of the maximum singular value

        if min(S_normalized) < self.gamma:
            projection = np.dot(error_vector, U[:, 0]) * U[:, 0]
            projected_target = current_pos + projection
            self.last_projection = (error_vector, U[:, 0], projection, projected_target)
            return projected_target, True

        self.last_projection = None
        return target, False


    def draw_arrow(self, start_pos, direction, color, scale=30):
        end_pos = (start_pos[0] + direction[0] * scale, 
                  start_pos[1] - direction[1] * scale)
        
        pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
        
        angle = math.atan2(-direction[1], direction[0])
        arr_len = 10
        arr_angle = math.pi/6
        
        arr1 = (end_pos[0] - arr_len * math.cos(angle + arr_angle),
                end_pos[1] - arr_len * math.sin(angle + arr_angle))
        arr2 = (end_pos[0] - arr_len * math.cos(angle - arr_angle),
                end_pos[1] - arr_len * math.sin(angle - arr_angle))
        
        pygame.draw.polygon(self.screen, color, [end_pos, arr1, arr2])

    def draw_button(self):
        button_rect = pygame.Rect(850, 600, 180, 50)
        mouse_pos = pygame.mouse.get_pos()
        color = self.BUTTON_HOVER_COLOR if button_rect.collidepoint(mouse_pos) else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, color, button_rect)
        button_text = "Projection: ON" if self.projection_mode else "Projection: OFF"
        text_surface = self.font.render(button_text, True, self.BUTTON_TEXT_COLOR)
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)
        return button_rect


    def draw_slider(self):
        slider_rect = pygame.Rect(850, 500, 300, 20)
        pygame.draw.rect(self.screen, self.SLIDER_COLOR, slider_rect)
        knob_x = 850 + (self.SINGULARITY_THRESHOLD / 100) * 300
        knob_rect = pygame.Rect(knob_x - 10, 495, 20, 30)
        pygame.draw.rect(self.screen, self.SLIDER_KNOB_COLOR, knob_rect)
        slider_text = f"Singularity Threshold: {self.gamma}"
        slider_surface = self.font.render(slider_text, True, self.BLACK)
        self.screen.blit(slider_surface, (850, 460))
        return slider_rect, knob_rect
    
    def draw_slider_ori(self):
        slider_rect = pygame.Rect(850, 400, 300, 20)
        pygame.draw.rect(self.screen, self.SLIDER_COLOR, slider_rect)
        knob_x = 1000 + (self.target_theta / (np.pi)) * 150
        knob_rect = pygame.Rect(knob_x - 10, 395, 20, 30)
        pygame.draw.rect(self.screen, self.SLIDER_KNOB_COLOR, knob_rect)
        slider_text = f"Target theta: {self.target_theta}"
        slider_surface = self.font.render(slider_text, True, self.BLACK)
        self.screen.blit(slider_surface, (850, 430))
        return slider_rect, knob_rect

    def handle_slider_input(self, slider_rect, knob_rect):
        mouse_pos = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0] and knob_rect.collidepoint(mouse_pos):
            relative_x = mouse_pos[0] - slider_rect.x
            relative_x = max(0, min(slider_rect.width, relative_x))
            self.SINGULARITY_THRESHOLD = int((relative_x / slider_rect.width) * 100)
            self.gamma = self.SINGULARITY_THRESHOLD / 100

    def handle_slider_input2(self, slider_rect, knob_rect):
        mouse_pos = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0] and knob_rect.collidepoint(mouse_pos): 
            relative_x = mouse_pos[0] - slider_rect.x
            relative_x = max(0, min(slider_rect.width, relative_x))
            target_theta = int((((relative_x / slider_rect.width) * 2 * np.pi) - np.pi) * 100)
            self.target_theta = target_theta / 100

    def draw(self):
        self.screen.fill(self.WHITE)
        
        pygame.draw.line(self.screen, self.GRAY, (0, self.offset_y), (self.offset_x * 2, self.offset_y))
        pygame.draw.line(self.screen, self.GRAY, (self.offset_x, 0), (self.offset_x, self.height))
        
        positions = self.forward_kinematics()
        
        for i in range(len(positions)-1):
            start = (positions[i][0] + self.offset_x, -positions[i][1] + self.offset_y)
            end = (positions[i+1][0] + self.offset_x, -positions[i+1][1] + self.offset_y)
            pygame.draw.line(self.screen, self.BLACK, start, end, 4)
            
            # Draw the angle of each joint
            joint_angle_deg = np.degrees(self.angles[i]) % 360
            if joint_angle_deg > 180:
                joint_angle_deg -= 360
            angle_text = f"∡{joint_angle_deg:.1f}°"
            angle_surface = self.small_font.render(angle_text, True, self.RED)
            angle_pos = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            self.screen.blit(angle_surface, angle_pos)
        
        for pos in positions:
            pygame.draw.circle(self.screen, self.BLUE, (int(pos[0] + self.offset_x), int(-pos[1] + self.offset_y)), 5)
        
        pygame.draw.circle(self.screen, self.MAGENTA, (int(self.target[0] + self.offset_x), int(-self.target[1] + self.offset_y)), 5)
        target_text = f"Target: ({self.target[0]:.2f}, {self.target[1]:.2f})"
        target_surface = self.small_font.render(target_text, True, self.MAGENTA)
        self.screen.blit(target_surface, (10, 150))
        
        if self.projected_target is not None:
            pygame.draw.circle(self.screen, self.ORANGE, (int(self.projected_target[0] + self.offset_x), int(-self.projected_target[1] + self.offset_y)), 5)
            pygame.draw.line(self.screen, self.ORANGE, (int(self.target[0] + self.offset_x), int(-self.target[1] + self.offset_y)), (int(self.projected_target[0] + self.offset_x), int(-self.projected_target[1] + self.offset_y)), 2)
            projected_text = f"Projected: ({self.projected_target[0]:.2f}, {self.projected_target[1]:.2f})"
            projected_surface = self.small_font.render(projected_text, True, self.ORANGE)
            self.screen.blit(projected_surface, (10, 170))
        
        end_effector_pos = (positions[-1][0] + self.offset_x, -positions[-1][1] + self.offset_y)
        pygame.draw.circle(self.screen, self.CYAN, (int(end_effector_pos[0]), int(end_effector_pos[1])), 5)
        current_pos_text = f"Current: ({positions[-1][0]:.2f}, {positions[-1][1]:.2f})"
        current_pos_surface = self.small_font.render(current_pos_text, True, self.CYAN)
        self.screen.blit(current_pos_surface, (10, 190))
        
        J = self.calculate_jacobian()
        U, S, Vt = svd(J)
        
        self.draw_arrow(end_effector_pos, U[:, 0], self.BLUE, scale=S[0] * 10)
        self.draw_arrow(end_effector_pos, U[:, 1], self.GREEN, scale=S[1] * 10)
        
        if self.show_jacobian:
            J_text = [
                f"J = [{J[0,0]:6.3f} {J[0,1]:6.3f} {J[0,2]:6.3f}]",
                f"    [{J[1,0]:6.3f} {J[1,1]:6.3f} {J[1,2]:6.3f}]"
            ]
            
            for i, text in enumerate(J_text):
                text_surface = self.small_font.render(text, True, self.BLACK)
                self.screen.blit(text_surface, (10, 10 + i * 20))
            
            s_text = f"Singular values: [{S[0]:6.3f}, {S[1]:6.3f}]"
            s_surface = self.small_font.render(s_text, True, self.BLACK)
            self.screen.blit(s_surface, (10, 50))
            
            u_text_1 = f"U[:, 0] (Blue): [{U[0,0]:6.3f}, {U[1,0]:6.3f}]"
            u_text_2 = f"U[:, 1] (Green): [{U[0,1]:6.3f}, {U[1,1]:6.3f}]"
            u_surface_1 = self.small_font.render(u_text_1, True, self.BLUE)
            u_surface_2 = self.small_font.render(u_text_2, True, self.GREEN)
            self.screen.blit(u_surface_1, (10, 80))
            self.screen.blit(u_surface_2, (10, 100))
            
            if self.projected_target is not None:
                proj_text = "Target projected due to singularity"
                proj_surface = self.small_font.render(proj_text, True, self.ORANGE)
                self.screen.blit(proj_surface, (10, 120))
                
                if self.last_projection is not None:
                    error_vector, u_vector, projection, projected_target = self.last_projection
                    projection_info = [
                        "Projection Calculations:",
                        f"Error Vector: [{error_vector[0]:.2f}, {error_vector[1]:.2f}]",
                        f"Projection Direction (U[:, 0]): [{u_vector[0]:.2f}, {u_vector[1]:.2f}]",
                        f"Projection: [{projection[0]:.2f}, {projection[1]:.2f}]",
                        f"Projected Target: [{projected_target[0]:.2f}, {projected_target[1]:.2f}]"
                    ]
                    
                    for i, line in enumerate(projection_info):
                        info_surface = self.small_font.render(line, True, self.BLACK)
                        self.screen.blit(info_surface, (850, 300 + i * 20))
        
        self.draw_button()
        slider_rect, knob_rect = self.draw_slider()
        slider_rect2, knob_rect2 = self.draw_slider_ori()
        self.handle_slider_input2(slider_rect2, knob_rect2)
        self.handle_slider_input(slider_rect, knob_rect)
        
        pygame.display.flip()

    def handle_mouse_input(self):
        mouse_pos = pygame.mouse.get_pos()
        mouse_x = mouse_pos[0] - self.offset_x
        mouse_y = -(mouse_pos[1] - self.offset_y)
        
        if pygame.mouse.get_pressed()[0]:
            button_rect = pygame.Rect(850, 600, 180, 50)
            if button_rect.collidepoint(mouse_pos):
                self.projection_mode = not self.projection_mode
                pygame.time.wait(150)
            else:
                if mouse_pos[0] < 800:  # Restrict target selection to the workspace area
                    self.target = np.array([mouse_x, mouse_y, self.target_theta])
                    self.projected_target = None
                    self.is_animating = True
                

    def adjusted_condition_number(self, J):
        """
        Calculate the adjusted condition number of the Jacobian matrix
        """
        U, S, Vt = svd(J) #obtain the singular value decomposition of the Jacobian matrix
        sigma_max = max(S) #find the maximum singular value
        S_normalized = [sig/ sigma_max for sig in S] #normalize the singular values with respect of the maximum singular value
        return S_normalized



    def parsed_jacobian(self, J):
        """
        Calculate the parsed Jacobian matrix
        """
        # Perform the singular value decomposition of the Jacobian matrix
        U, S, Vt = svd(J) #obtain the singular value decomposition of the Jacobian matrix
        # Rebuild U and S with columns that are strictly greater than the threshold
        U_new = []
        S_new = []
        sigma_max = max(S)
        for col in range(len(S)):
            if S[col] > self.gamma * sigma_max:
                #Singular row
                U_new.append(np.matrix(U[:,col]).T)
                S_new.append(S[col])
        U_new = np.concatenate(U_new,axis=1)
        J_parsed = self.svd_compose(U_new, S_new, Vt)
        return J_parsed


    def stable_jacobian(self, J):
        """
        Calculate the stable Jacobian matrix
        """
        # Perform the singular value decomposition of the Jacobian matrix
        U, S, Vt = svd(J) #obtain the singular value decomposition of the Jacobian matrix
        s_max = np.max(S)
        S_new = [s if (s/s_max) > self.gamma else self.gamma*s_max for s in S]
        J_stable = self.svd_compose(U,S_new,Vt)
        return J_stable


    def singular_projection(self, J, adjusted_condition_numbers):
        """
        Calculate the singular projection matrix, this is the scaled projection
        of the task-space vector onto the singular directions
        """
        # Perform the singular value decomposition of the Jacobian matrix
        U, S, Vt = svd(J) #obtain the singular value decomposition of the Jacobian matrix 
        #Generate a new U that only has singular directions
        U_new = []
        Phi = [] #these will be the ratio of s_i/s_max
        set_empty_bool = True
        for col in range(len(S)):
            if adjusted_condition_numbers[col] <= self.gamma:
                set_empty_bool = False
                U_new.append(np.matrix(U[:,col]).T)
                Phi.append(adjusted_condition_numbers[col])

        #set an empty Phi_singular matrix, populate if there were any adjusted
        #condition numbers below the threshold
        Phi_singular = np.zeros(U.shape) #initialize the singular projection matrix  

        if set_empty_bool == False:
            #construct the new U, as there were singular directions
            U_new = np.matrix(np.concatenate(U_new,axis=1))
            Phi_mat = np.matrix(np.diag(Phi))
            # Now put it all together:
            Phi_singular = U_new @ Phi_mat @ U_new.T
         
        return Phi_singular, set_empty_bool


    def svd_compose(self,U,S,Vt):
        """
        This function takes SVD: U,S,V and recomposes them for J
        """
        Zero_concat = np.zeros((U.shape[0],Vt.shape[0]-len(S)))
        Sfull = np.zeros((U.shape[1],Vt.shape[0]))
        for row in range(Sfull.shape[0]):
            for col in range(Sfull.shape[1]):
              if row == col:
                  if row < len(S):        
                      Sfull[row,col] = S[row]
        J_new =np.matrix(U)*Sfull*np.matrix(Vt)
        return J_new


    def inverse_kinematics_step(self):
        if not self.is_animating:
            return
            
        current_pos = np.array(self.forward_kinematics()[-1])
        J = self.calculate_jacobian()
        U, S, Vt = svd(J)
        current_end_pose = np.concatenate((current_pos, np.array([np.sum(self.angles)])))
        target_to_use = self.target

        if self.projection_mode:
            e_proj, was_projected = self.project_target_on_valid_direction(current_end_pose, self.target, U, S)
            if was_projected:
                self.projected_target = e_proj
        
        error = target_to_use - current_end_pose
        error *= [1, 1, 1000000]
        if np.linalg.norm(error) < 1.0:
            #self.is_animating = False
            return
            
        #### This is where the magic happens
        # we calculate the adjusted condition number for each singular value of the Jacobian matrix
        adjusted_condition_numbers = self.adjusted_condition_number(J)
        # We calculate the Parsed Jacobian matrix
        J_parsed = self.parsed_jacobian(J)
        # We calculate the Stable Jacobian matrix
        J_stable = self.stable_jacobian(J)
        # We calculate the projection of the error vector in singular directions scaled by the relative adjusted condition numbers
        Phi_singular, set_empty_bool = self.singular_projection(J,adjusted_condition_numbers)  

        J_stable_inv = np.linalg.pinv(J_stable)
        J_parsed_inv = np.linalg.pinv(J_parsed)
        if set_empty_bool == False:
            # in a singular configuration
            
            delta_theta = J_stable_inv @ J_parsed @ J_parsed_inv @ error + J_stable_inv @ Phi_singular @ error 
        else:
            # not in a singular config 
            delta_theta = J_stable_inv @ J_parsed @ J_parsed_inv @ error #the traditional J_inv can also be used here (less work, but same thing)
            # delta_theta = J_inv @ error #same thing
            
        delta_theta = np.squeeze(np.asarray(delta_theta))
        self.angles += delta_theta * 0.005
        #self.angles = np.clip(self.angles, -np.pi, np.pi)
        self.end_theta = np.sum(self.angles)
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_j:
                        self.show_jacobian = not self.show_jacobian
            
            self.handle_mouse_input()
            self.inverse_kinematics_step()
            self.draw()
            self.clock.tick(500)
            
        pygame.quit()

if __name__ == "__main__":
    arm = RoboticArm()
    arm.run()
