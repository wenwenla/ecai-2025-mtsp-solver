cmake_minimum_required(VERSION 3.12)
project(splitting_solver)

find_package(pybind11 REQUIRED PATHS /usr/local/lib/python3.10/dist-packages/pybind11)

pybind11_add_module(splitting_solver solver.cpp)