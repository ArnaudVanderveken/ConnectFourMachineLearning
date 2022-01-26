#include "pch.h"
#include "AI.h"
#include "Grid.h"

using Eigen::Rand::normal, Eigen::Matrix, std::vector, std::ofstream, std::ifstream;

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
	m_Weights = normal<Matrix<float, 42, 1>>(42, 1, gen, 0.0f, 0.001f);
	m_Trace.fill(0);
}

AI::AI(string filename)
{
	m_AILearning = AILearning::QLearning;
	m_Epsilon = 0;
	m_LearningRate = 0;
	m_Lambda = 0;

	char* buffer = new char[sizeof(float) * m_Weights.rows() * m_Weights.cols()];

	ifstream input;
	input.open(filename, std::ios::binary);

	input.read(buffer, sizeof(int));
	m_AILearning = (AILearning(atoi(buffer)));

	input.read(buffer, sizeof(float));
	m_Epsilon = (float)atof(buffer);

	if (m_AILearning == AILearning::TDLambda)
	{
		input.read(buffer, sizeof(float));
		m_Lambda = (float)atof(buffer);
	}

	float* matrixData = new float[m_Weights.rows() * m_Weights.cols()];
	input.read((char*)matrixData, sizeof(float) * m_Weights.rows() * m_Weights.cols());
	m_Weights = Matrix<float, 42, 1>(matrixData);

	if (m_AILearning == AILearning::TDLambda)
	{
		input.read((char*)matrixData, sizeof(float) * m_Trace.rows() * m_Trace.cols());
		m_Trace = Matrix<float, 42, 1>(matrixData);
	}

	delete[] buffer;
	delete[] matrixData;
	input.close();
}

int AI::PlayMove(Grid* pGrid, bool asPlayer1, bool trainingMode)
{
	int playedMove{}, bestMove{};

	//Find the best move acording to the NN
	vector<float> probabilities;
	probabilities.reserve(7);
	Matrix<float, 1, 42> gridStateSavePre{ pGrid->GetStateMatrix() };
	Matrix<float, 1, 42> gridStateSavePost{ pGrid->GetStateMatrix() };

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
			gridStateSavePost(0, pGrid->GetAvailableRowInColumn(bestMove) * Grid::s_NrColumns + bestMove) = asPlayer1 ? 1.0f : -1.0f;
			NNQLearning(gridStateSavePre, gridStateSavePost);
			break;

		case AILearning::TDLambda:
			gridStateSavePost(0, pGrid->GetAvailableRowInColumn(playedMove) * Grid::s_NrColumns + playedMove) = asPlayer1 ? 1.0f : -1.0f;
			NNTDLambda(gridStateSavePre, gridStateSavePost);
			break;
		}
	}

	return playedMove;
}

void AI::SaveToFile(string filename)
{
	ofstream output{};
	output.open(filename, std::ios::binary);

	int AILearningMethod{ int(m_AILearning) };
	output.write((const char *)&AILearningMethod, sizeof(int));
	output.write((const char*)&m_Epsilon, sizeof(float));
	output.write((const char*)&m_LearningRate, sizeof(float));
	if (m_AILearning == AILearning::TDLambda) output.write((const char*)&m_Lambda, sizeof(float));
	output.write((const char*)m_Weights.data(), sizeof(float) * (uint32_t)m_Weights.rows() * (uint32_t)m_Weights.cols());
	if (m_AILearning == AILearning::TDLambda)output.write((const char*)m_Trace.data(), sizeof(float) * (uint32_t)m_Trace.rows() * (uint32_t)m_Trace.cols());

	output.close();
}

float AI::NNForwardPass(const Matrix<float, 1, 42>& input) const
{
	return Sigmoid(input * m_Weights);
}

void AI::NNQLearning(const Matrix<float, 1, 42>& oldState, const Matrix<float, 1, 42>& bestState)
{
	float oldStatePOut{ NNForwardPass(oldState) };
	float delta{ NNForwardPass(bestState) - oldStatePOut };
	float grad{ oldStatePOut * (1 - oldStatePOut) }; //Sigmoid derivative.

	for (int i{}; i < 42; ++i)
	{
		m_Weights(i, 0) += m_LearningRate * delta * grad * oldState(0, i);
	}
}

void AI::NNQLearningFinal(const Eigen::Matrix<float, 1, 42>& oldState, float result)
{
	float delta{ result - NNForwardPass(oldState) };

	for (int i{}; i < 42; ++i)
	{
		m_Weights(i, 0) += m_LearningRate * delta * oldState(0, i);
	}
}

void AI::NNTDLambda(const Matrix<float, 1, 42>& oldState, const Matrix<float, 1, 42>& playedState)
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
