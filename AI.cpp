#include "pch.h"
#include "AI.h"

using Eigen::Rand::normal, Eigen::Matrix;

AI::AI(float epsilon, float learningRate, float lambda)
	: m_Epsilon{ std::clamp(epsilon, 0.0f, 1.0f) }
	, m_LearningRate{ std::clamp(learningRate, 0.0f, 1.0f) }
	, m_Lambda{ std::clamp(lambda, 0.0f, 1.0f) }
{
	// Initializing weights based on a normal distribution (mean 0.0, stdev 0.0001)
	Eigen::Rand::P8_mt19937_64 urng{ 42 };
	m_Wint = normal<Matrix<float, 84, s_InnerLayerNeuronCount>>(84, s_InnerLayerNeuronCount, urng, 0.0f, 0.0001f);
	m_Wout = normal<Matrix<float, s_InnerLayerNeuronCount, 1>>(s_InnerLayerNeuronCount, 1, urng, 0.0f, 0.0001f);
}

float AI::NNForwardPass(Matrix<float, 1, 84> input) const
{
	Matrix<float, 1, s_InnerLayerNeuronCount> intermediate{ input * m_Wint };
	Sigmoid(intermediate);
	float out = intermediate * m_Wout;
	return Sigmoid(out);
}

void AI::NNQLearning(Matrix<float, 1, 84> oldState, Matrix<float, 1, 84> bestState)
{
	Matrix<float, 1, s_InnerLayerNeuronCount> intermediate{ bestState * m_Wint };
	Matrix<float, 1, s_InnerLayerNeuronCount> bestStatePInt = Sigmoid(intermediate);
	float bestStatePOut{ NNForwardPass(bestState) };
	float delta{ NNForwardPass(oldState) - bestStatePOut };
	float gradOut{ bestStatePOut * (1 - bestStatePOut) };

	Matrix<float, 1, s_InnerLayerNeuronCount> gradInt;
	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		gradInt[0, i] = bestStatePInt[0, i] * (1 - bestStatePInt[0, i]);
	}

	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		float deltaInt = gradOut * m_Wout[0, i] * gradInt[0, i];
		for (int j{}; j < 84; ++j)
		{
			m_Wint[j, i] -= m_LearningRate * delta * deltaInt * bestState[0, j];
		}
		m_Wout[i, 0] -= m_LearningRate * delta * gradOut * bestStatePInt[0, i];
	}
}

void AI::NNTDLambda(Matrix<float, 1, 84> oldState, Matrix<float, 1, 84> playedState)
{
}

float AI::Sigmoid(float x) const
{
	return 1.0f / (1.0f - powf(2.71828f, x));
}

const Matrix<float, 1, AI::s_InnerLayerNeuronCount>& AI::Sigmoid(Matrix<float, 1, s_InnerLayerNeuronCount>& m) const
{
	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		m[0, i] = Sigmoid(m[0, i]);
	}
	return m;
}
