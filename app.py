from numpy._typing import NDArray
import pygame
import numpy as np
import numpy.typing as npt
from prettytable import PrettyTable
import time

from abc import ABC, abstractmethod

class ODE(ABC):
    def __init__(self, eqns_num: np.int_) -> None:
        assert eqns_num > 0

        self.eqns_num = eqns_num
        self.q_arr = np.empty(eqns_num, dtype=np.single)
        self.s = np.single(0)

    def get_q(self, index: np.int_) -> np.single:
        assert index >= 0
        assert index < self.eqns_num
        assert self.q_arr.size == self.eqns_num

        return np.single(self.q_arr[index])
    
    def set_q(self, index: np.int_, value: np.single) -> None:
        assert index >= 0
        assert index < self.eqns_num
        assert self.q_arr.size == self.eqns_num

        self.q_arr[index] = value

    @abstractmethod
    def dq_arr(
        self, 
        s: np.single, 
        q_arr: npt.NDArray[np.single],
        prev_dq_arr: npt.NDArray[np.single],
        ds: np.single,
        q_scale: np.single,
    ) -> npt.NDArray[np.single]:
        pass


# Fourth-order Runge-Kutta ODE Solver
def solve_ODE(ode: ODE, ds: np.single):
    eqns_num = ode.eqns_num
    q_arr = ode.q_arr
    s = ode.s

    dq_arr1 = ode.dq_arr(s, q_arr, q_arr, ds, 0.5)
    assert dq_arr1.size == eqns_num
    dq_arr2 = ode.dq_arr(s + (0.5 * ds), q_arr, dq_arr1, ds, 0.5)
    assert dq_arr2.size == eqns_num
    dq_arr3 = ode.dq_arr(s + (0.5 * ds), q_arr, dq_arr2, ds, 0.5)
    assert dq_arr3.size == eqns_num
    dq_arr4 = ode.dq_arr(s + ds, q_arr, dq_arr3, ds, 1)
    assert dq_arr4.size == eqns_num

    ode.s = s + ds

    for i in range(eqns_num):
        q_arr[i] = q_arr[i] + (
                dq_arr1[i] + 
                (2.0 * dq_arr2[i]) + 
                (2.0 * dq_arr3[i]) + 
                dq_arr4[i]
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

        self.set_q(0, 0.0)
        self.set_q(1, x0)

    def velocity(self) -> np.single:
        return self.get_q(0)
    
    def position(self) -> np.single:
        return self.get_q(1)

    def time(self) -> np.single:
        return self.s
    
    def update(self, dt: np.single) -> None:
        solve_ODE(self, dt)

    def dq_arr(
        self, 
        s: np.single, 
        q_arr: npt.NDArray[np.single],
        prev_dq_arr: npt.NDArray[np.single],
        ds: np.single,
        q_scale: np.single,
    ) -> npt.NDArray[np.single]:
        dq_arr = np.empty(self.eqns_num, dtype=np.single)
        new_q_arr = np.empty(self.eqns_num, dtype=np.single)

        for i in range(self.eqns_num):
            new_q_arr[i] = q_arr[i] + (q_scale * prev_dq_arr[i])
        
        dq_arr[0] = -ds * (
                (self.mu * new_q_arr[0]) + (self.k * new_q_arr[1])
            ) / self.mass
        dq_arr[1] = ds * new_q_arr[0]

        return dq_arr
    
def ticks_in_seconds() -> float:
    return float(pygame.time.get_ticks()) / 1000

if __name__ == "__main__":
    print("Hello, World!")

    spring_ode = SpringODE(1.0, 1.5, 20.0, -0.2)
    
    printable = True
    table = PrettyTable()

    table.field_names = ["time", "position", "velocity"]
    table.add_row([
        spring_ode.time(), 
        spring_ode.position(), 
        spring_ode.velocity(),
    ])  

    # pygame setup
    pygame.init()
    icon_img = pygame.image.load("icon.png")
    pygame.display.set_icon(icon_img)
    pygame.display.set_caption("Spring Motion Simulator")
    screen = pygame.display.set_mode((500, 500))
    clock = pygame.time.Clock()
    running = True

    total_sec: float = 7.0
    prev_sec: float
    curr_sec: float = ticks_in_seconds()
    dt: float

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # RENDER YOUR GAME HERE
        prev_sec = curr_sec
        curr_sec = ticks_in_seconds()
        dt = curr_sec - prev_sec
        if printable == True:
            print(f"dt: {dt}")

        spring_ode.update(dt)

        x: int = int(spring_ode.position() * 1000) + 250  # TODO: refactoring
        pygame.draw.circle(screen, (200, 200, 200), [x, 250], 10)

        table.add_row([
            spring_ode.time(), 
            spring_ode.position(), 
            spring_ode.velocity(),
        ])

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

        if spring_ode.time() > total_sec and printable == True:
            print(table)
            printable = False

    pygame.quit()
    