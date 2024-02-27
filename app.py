import pygame

import numpy as np
import numpy.typing as npt

from prettytable import PrettyTable

from abc import ABC, abstractmethod

class ODE(ABC):
    def __init__(self, eqns_num: np.int_) -> None:
        assert eqns_num > 0

        self.EQNS_NUM = eqns_num
        self.q = np.empty(eqns_num, dtype=np.single)
        self.t = np.single(0)

    def get_value(self, index: np.int_) -> np.single:
        assert index >= 0
        assert index < self.EQNS_NUM
        assert self.q.size == self.EQNS_NUM

        return np.single(self.q[index])
    
    def set_value(self, index: np.int_, value: np.single) -> None:
        assert index >= 0
        assert index < self.EQNS_NUM
        assert self.q.size == self.EQNS_NUM

        self.q[index] = value

    @abstractmethod
    def dq(
        self, 
        t: np.single, 
        q: npt.NDArray[np.single],
        dq_prev: npt.NDArray[np.single],
        dt: np.single,
        q_scale: np.single,
    ) -> npt.NDArray[np.single]:
        pass


# Fourth-order Runge-Kutta ODE Solver
def solve_ODE(ode: ODE, dt: np.single):
    eqns_num = ode.EQNS_NUM
    q = ode.q
    t = ode.t

    dq1 = ode.dq(t, q, q, dt, 0.5)
    assert dq1.size == eqns_num
    dq2 = ode.dq(t + (0.5 * dt), q, dq1, dt, 0.5)
    assert dq2.size == eqns_num
    dq3 = ode.dq(t + (0.5 * dt), q, dq2, dt, 0.5)
    assert dq3.size == eqns_num
    dq4 = ode.dq(t + dt, q, dq3, dt, 1)
    assert dq4.size == eqns_num

    ode.t = t + dt

    for i in range(eqns_num):
        q[i] = q[i] + (
                dq1[i] + 
                (2.0 * dq2[i]) + 
                (2.0 * dq3[i]) + 
                dq4[i]
            ) / 6.0
    
    return

class SpringODE(ODE):
    def __init__(
        self, 
        mass: np.single, mu: np.single, 
        k: np.single, x0: np.single,
    ) -> None:
        super().__init__(2)

        self.mass = mass
        self.mu = mu
        self.k = k
        self.x0 = x0

        self.set_value(0, 0.0)
        self.set_value(1, x0)

    def velocity(self) -> np.single:
        return self.get_value(0)
    
    def position(self) -> np.single:
        return self.get_value(1)

    def time(self) -> np.single:
        return self.t
    
    def update(self, dt: np.single) -> None:
        solve_ODE(self, dt)

    def dq(
        self, 
        s: np.single, 
        q: npt.NDArray[np.single],
        dq_prev: npt.NDArray[np.single],
        dt: np.single,
        q_scale: np.single,
    ) -> npt.NDArray[np.single]:
        dq = np.empty(self.EQNS_NUM, dtype=np.single)
        new_q = np.empty(self.EQNS_NUM, dtype=np.single)

        for i in range(self.EQNS_NUM):
            new_q[i] = q[i] + (q_scale * dq_prev[i])
        
        dq[0] = -dt * (
                (self.mu * new_q[0]) + (self.k * new_q[1])
            ) / self.mass
        dq[1] = dt * new_q[0]

        return dq
    
def ticks_in_seconds() -> np.single:
    return np.single(pygame.time.get_ticks()) / 1000

def to_x_screen(
        x0: np.single, 
        screen_width: np.int_, 
        screen_margin: np.int_,
        x: np.single,
    ) -> np.int_:
    x0_prime = abs(x0)
    assert x0_prime > 0
    assert screen_width > 0
    assert screen_margin > 0

    a = np.single(screen_width - (screen_margin * 2)) / (x0_prime * 2)
    x_screen: np.int_ = np.int_((x * a) + (screen_width / 2))

    return x_screen

FRAMERATE: np.single = 30

PRINTABLE = False

SCREEN_WIDTH: np.int_ = 500
SCREEN_HEIGHT: np.int_ = 500
SCREEN_MARGIN: np.int_ = 50

MASS: np.single = 1.0
MU: np.single = 1.5  # the damping coefficient
K: np.single = 20.0  # the spring constant
X0: np.single = -0.2

END_TIME: np.single = 7.0  # in seconds

if __name__ == "__main__":
    print("Hello, World!")

    intro_table = PrettyTable()
    intro_table.add_column(
        fieldname="Name", 
        column=[
            "Mass at end of Spring (kg)", 
            "Damping Coefficient (kg/s)", 
            "Spring Constant (N/m)", 
            "Initial Spring Deflection (m)",
            "End Time to Record (s)",
        ]
    )
    intro_table.add_column(
        fieldname="Value", 
        column=[MASS, MU, K, X0, END_TIME])
    print(intro_table)

    spring_ode = SpringODE(MASS, MU, K, X0)

    finish = False
    result_table = PrettyTable(["Time", "Position", "Velocity"])
    
    result_table.add_row([
        spring_ode.time(), 
        spring_ode.position(), 
        spring_ode.velocity(),
    ])  

    # pygame setup
    pygame.init()
    icon_img = pygame.image.load("icon.png")
    pygame.display.set_icon(icon_img)
    pygame.display.set_caption("Spring Motion Simulator")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    dt: np.single = 1.0/FRAMERATE

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # RENDER YOUR GAME HERE

        spring_ode.update(dt)

        if PRINTABLE == True and finish == False:
            print(f"t: {spring_ode.time()}")

        x_screen: np.int_ = to_x_screen(
                X0, 
                SCREEN_WIDTH, SCREEN_MARGIN,
                spring_ode.position(),
            )
        pygame.draw.circle(
            screen, 
            (200, 200, 200), 
            [x_screen, 250], 
            20)
    
        if finish == False:
            result_table.add_row([
                spring_ode.time(), 
                spring_ode.position(), 
                spring_ode.velocity(),
            ])

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(FRAMERATE)  # limits FPS to FRAMERATE

        if spring_ode.time() > END_TIME and finish == False:
            print(result_table)
            finish = True
            PRINTABLE = False

    pygame.quit()
    