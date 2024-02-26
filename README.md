# mancala-parallel-mcts
Parallel UCT-MCTS AI for the Mancala (Kalaha) board game.

mancala.cpp is the default version.

Compile using C++ compiler of your choice. I recommend using at least C++17 and to use full optimizations `-O3`.

`clang++ -std=c++20 -O3 -o mancala mancala.cpp`

This project was meant as an exercise in C++ parallel programming techniques, it's not very ergonomic as a software.
In order to change the memory consumed etc. you need to modify the constants in the code.


