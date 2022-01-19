#include "pch.h"
#include "AI.h"

AI::AI()
{
	// Initializing weights based on a normal distribution (mean 0.0, stdev 0.0001)
	Eigen::Rand::P8_mt19937_64 urng{ 42 };
	m_Win = Eigen::Rand::normal<Eigen::Matrix<float, 84, s_InnerLayerNeuronCount>>(84, s_InnerLayerNeuronCount, urng, 0.0f, 0.0001f);
	m_Wout = Eigen::Rand::normal<Eigen::Matrix<float, s_InnerLayerNeuronCount, 1>>(s_InnerLayerNeuronCount, 1, urng, 0.0f, 0.0001f);
}
