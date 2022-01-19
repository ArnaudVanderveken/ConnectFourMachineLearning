#include "pch.h"
#include "AI.h"

using Eigen::Rand::normal, Eigen::Matrix;

AI::AI()
{
	// Initializing weights based on a normal distribution (mean 0.0, stdev 0.0001)
	Eigen::Rand::P8_mt19937_64 urng{ 42 };
	m_Win = normal<Matrix<float, 84, s_InnerLayerNeuronCount>>(84, s_InnerLayerNeuronCount, urng, 0.0f, 0.0001f);
	m_Wout = normal<Matrix<float, s_InnerLayerNeuronCount, 1>>(s_InnerLayerNeuronCount, 1, urng, 0.0f, 0.0001f);
}

float AI::NNForwardPass(Eigen::Matrix<float, 1, 84> input) const
{
	Matrix<float, 1, s_InnerLayerNeuronCount> intermediate = input * m_Win;
	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		intermediate[0, i] = Sigmoid(intermediate[0, i]);
	}
	float out = intermediate * m_Wout;
	return Sigmoid(out);
}

float AI::Sigmoid(float x) const
{
	return 1.0f / (1.0f - powf(2.71828f, x));
}
