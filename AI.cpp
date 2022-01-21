#include "pch.h"
#include "AI.h"
#include "Grid.h"

using Eigen::Rand::normal, Eigen::Matrix, std::vector;

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

AI::AI(string filename)
{
	m_Epsilon = 0;
	m_LearningRate = 0;
	m_Lambda = 0;
}

int AI::PlayMove(Grid* pGrid, bool asPlayer1, bool trainingMode)
{
	if (trainingMode) //Only do E-Greedy test in trainingMode. For an actual game, always play best move.
	{
		//E-Greedy test. If value is below epsilon, play random move.
		float greedyTest{ rand() / float(RAND_MAX) }; //float btwn 0.0f - 1.0f
		if (greedyTest < m_Epsilon)
			return (rand() % 7); //int btwn 0 - 6
	}

	//Play the best move acording to the NN
	vector<float> probabilities;
	probabilities.reserve(7);
	Matrix<float, 1, 84> gridState{ pGrid->GetStateMatrix() }; //gridState always puts P1 in the first half and P2 in the second half of the matrix
	Matrix<float, 1, 84> playState;

	//Swap first and second halves of the gridState matrix if playing as P2
	if (asPlayer1)
		playState = gridState;
	else
	{
		playState.block(0, 0, 1, 42) = gridState.block(0, 42, 1, 42);
		playState.block(0, 42, 1, 42) = gridState.block(0, 0, 1, 42);
	}

	for (int i{}; i < 7; ++i)
	{
		int row{ pGrid->GetAvailableRowInColumn(i) };
		if (row == -1)
			probabilities.push_back(-1.0f);
		else
		{
			playState(0, row * 7 + i) = 1.0f;
			probabilities.push_back(NNForwardPass(playState));
			playState(0, row * 7 + i) = 0.0f;
		}
	}
	auto it = std::max_element(probabilities.begin(), probabilities.end());
	return (it - probabilities.begin()); //returns the column where to play (range 0 - 6)
}

void AI::SaveToFile(string filename)
{
	ofstream output{};
	output.open(filename);

	//TODO: write content to file
	//To write:
	// Learning method (0 - 1)
	// Epsilon
	// LearningRate
	// If TDLambda Lambda
	// Wint
	// Wout

	output.close();
}

float AI::NNForwardPass(const Matrix<float, 1, 84>& input) const
{
	Matrix<float, 1, s_InnerLayerNeuronCount> intermediate{ input * m_Wint };
	Sigmoid(intermediate);
	float out = intermediate * m_Wout;
	return Sigmoid(out);
}

void AI::NNQLearning(const Matrix<float, 1, 84>& oldState, const Matrix<float, 1, 84>& bestState)
{
	Matrix<float, 1, s_InnerLayerNeuronCount> intermediate{ bestState * m_Wint };
	Matrix<float, 1, s_InnerLayerNeuronCount> bestStatePInt = Sigmoid(intermediate);
	float bestStatePOut{ NNForwardPass(bestState) };
	float delta{ NNForwardPass(oldState) - bestStatePOut };
	float gradOut{ bestStatePOut * (1 - bestStatePOut) };

	Matrix<float, 1, s_InnerLayerNeuronCount> gradInt;
	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		gradInt(0, i) = bestStatePInt(0, i) * (1 - bestStatePInt(0, i));
	}

	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		float deltaInt = gradOut * m_Wout(0, i) * gradInt(0, i);
		for (int j{}; j < 84; ++j)
		{
			m_Wint(j, i) -= m_LearningRate * delta * deltaInt * bestState(0, j);
		}
		m_Wout(i, 0) -= m_LearningRate * delta * gradOut * bestStatePInt(0, i);
	}
}

void AI::NNTDLambda(const Matrix<float, 1, 84>& oldState, const Matrix<float, 1, 84>& playedState)
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
		m(0, i) = Sigmoid(m(0, i));
	}
	return m;
}
