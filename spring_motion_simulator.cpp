// spring_motion_simulator.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <SDL.h>
#undef main

int main(void)
{
    SDL_Init(SDL_INIT_VIDEO);

    std::cout << "Hello World!" << std::endl;
    
    return 0;
}
