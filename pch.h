// pch.h: This is a precompiled header file.
// Files listed below are compiled only once, improving build performance for future builds.
// This also affects IntelliSense performance, including code completion and many code browsing features.
// However, files listed here are ALL re-compiled if any one of them is updated between builds.
// Do not add files here that you will be updating frequently as this negates the performance advantage.

#ifndef PCH_H
#define PCH_H

// add headers that you want to pre-compile here
#include <Eigen/Core>

#pragma warning (push)
#pragma warning (disable : 4305) // disable warning caused by the EigenRand lib
#include <EigenRand/EigenRand>
#pragma warning (pop)

#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include <string>
#include <fstream>
#include <regex>
#include <math.h>

#include "Enums.h"

#define P1_TOKEN 'O'
#define P2_TOKEN 'X'
#define EMPTY_TOKEN '.'

#define WIN_CREDITS 1.0f
#define LOSS_CREDITS -1.0f
#define DRAW_CREDITS 0.0f

#endif //PCH_H
