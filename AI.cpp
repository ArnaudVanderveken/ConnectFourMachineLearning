#include "pch.h"
#include "AI.h"
#include "Grid.h"

using Eigen::Rand::normal, Eigen::Matrix, std::vector;

AI::AI(AILearning aiLearning, float epsilon, float learningRate, float lambda)
	: m_AILearning{ aiLearning }
	, m_Epsilon{ std::clamp(epsilon, 0.0f, 1.0f) }
	, m_LearningRate{ std::clamp(learningRate, 0.0f, 1.0f) }
	, m_Lambda{ std::clamp(lambda, 0.0f, 1.0f) }
{
	// Initializing weights based on a normal distribution (mean 0.0, stdev 0.001)
	//Eigen::Rand::P8_mt19937_64 urng{ 42 };
	std::random_device rd;
	std::mt19937 gen(rd());
	m_Weights = normal<Matrix<float, 84, 1>>(84, 1, gen, 0.0f, 0.001f);
}

AI::AI(string filename)
{
	m_AILearning = AILearning::QLearning;
	m_Epsilon = 0;
	m_LearningRate = 0;
	m_Lambda = 0;
}

int AI::PlayMove(Grid* pGrid, bool asPlayer1, bool trainingMode)
{
	int playedMove{}, bestMove{};

	//Find the best move acording to the NN
	vector<float> probabilities;
	probabilities.reserve(7);
	Matrix<float, 1, 84> gridStateSavePre{ pGrid->GetStateMatrix() };
	Matrix<float, 1, 84> gridStateSavePost{ pGrid->GetStateMatrix() };

	for (int i{}; i < 7; ++i)
	{
		int row{ pGrid->GetAvailableRowInColumn(i) };
		if (row == -1)
			probabilities.push_back(asPlayer1 ? -FLT_MAX : FLT_MAX);
		else
		{
			gridStateSavePre(0, row * 7 + i) = 1.0f;
			probabilities.push_back(NNForwardPass(gridStateSavePre));
			gridStateSavePre(0, row * 7 + i) = 0.0f;
		}
	}
	if (asPlayer1)
		bestMove = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
	else
		bestMove = std::min_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
	playedMove = bestMove; //Default

	// E-greedy & learning only in trainingmode
	if (trainingMode) //Only do E-Greedy test in trainingMode. For an actual game, always play best move.
	{
		//E-Greedy test. If value is below epsilon, play random move amongst possibles.
		float greedyTest{ rand() / float(RAND_MAX) }; //float btwn 0.0f - 1.0f
		if (greedyTest < m_Epsilon)
		{
			vector<int> possibleMoves;
			for (int i{}; i < Grid::s_NrColumns; ++i)
			{
				if (!pGrid->IsColumnFull(i)) possibleMoves.push_back(i);
			}
			playedMove = possibleMoves[rand() % possibleMoves.size()];
		}
		// else playedMove = bestMove, as default if no training

		switch (m_AILearning)
		{
		case AILearning::QLearning:
			gridStateSavePost(0, pGrid->GetAvailableRowInColumn(bestMove) * Grid::s_NrColumns + bestMove + (asPlayer1 ? 0 : 42)) = 1.0f;
			NNQLearning(gridStateSavePre, gridStateSavePost);
			break;

		case AILearning::TDLambda:
			gridStateSavePost(0, pGrid->GetAvailableRowInColumn(playedMove) * Grid::s_NrColumns + playedMove + (asPlayer1 ? 0 : 42)) = 1.0f;
			NNTDLambda(gridStateSavePre, gridStateSavePost);
			break;
		}
	}

	return playedMove;
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
	return std::tanhf(input * m_Weights);
}

void AI::NNQLearning(const Matrix<float, 1, 84>& oldState, const Matrix<float, 1, 84>& bestState)
{
	float oldStatePOut{ NNForwardPass(oldState) };
	float delta{ oldStatePOut - NNForwardPass(bestState) };

	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		m_Weights(i, 0) -= m_LearningRate * delta * oldState(0, i);
	}
}

void AI::NNQLearningFinal(const Eigen::Matrix<float, 1, 84>& oldState, float result)
{
	float delta{ NNForwardPass(oldState) - result };

	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		m_Weights(i, 0) -= m_LearningRate * delta * oldState(0, i);
	}
}

void AI::NNTDLambda(const Matrix<float, 1, 84>& oldState, const Matrix<float, 1, 84>& playedState)
{
}

float AI::Sigmoid(float x) const
{
	return 1.0f / (1.0f + exp(-x));
}

float AI::ReLU(float x) const
{
	return std::max(0.0f, x);
}

float AI::SWISH(float x) const
{
	return x * Sigmoid(x);
}
