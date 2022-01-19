#include "pch.h"
#include "AI.h"

AI::AI()
{
	Eigen::Rand::Vmt19937_64 generator;

	m_Win = Eigen::Rand::normal<Eigen::Matrix<float, 84, s_InnerLayerNeuronCount>>(84, s_InnerLayerNeuronCount, generator, 0.0f, 0.01f);
	m_Wout = Eigen::Rand::normal<Eigen::Matrix<float, s_InnerLayerNeuronCount, 1>>(s_InnerLayerNeuronCount, 1, generator, 0.0f, 0.01f);
}
