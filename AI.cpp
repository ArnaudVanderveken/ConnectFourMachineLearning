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
	m_Wint = normal<Matrix<float, 84, s_InnerLayerNeuronCount>>(84, s_InnerLayerNeuronCount, gen, 0.0f, 0.001f);
	m_Wout = normal<Matrix<float, s_InnerLayerNeuronCount, 1>>(s_InnerLayerNeuronCount, 1, gen, 0.0f, 0.001f);
	m_TraceInt.fill(0);
	m_TraceOut.fill(0);
}

AI::AI(string filename)
{
	m_AILearning = AILearning::QLearning;
	m_Epsilon = 0;
	m_LearningRate = 0;
	m_Lambda = 0;

	char* buffer = new char[sizeof(float)];

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

	float* matrixData = new float[m_Wint.rows() * m_Wint.cols()];
	input.read((char*)matrixData, sizeof(float) * m_Wint.rows() * m_Wint.cols());
	auto tmp = new Matrix<float, 84, 84>(matrixData); //Use of heap due to size
	m_Wint = *tmp;
	delete tmp;

	input.read((char*)matrixData, sizeof(float) * m_Wout.rows() * m_Wout.cols());
	m_Wout = Matrix<float, 84, 1>(matrixData);

	if (m_AILearning == AILearning::TDLambda)
	{
		input.read((char*)matrixData, sizeof(float) * m_TraceInt.rows() * m_TraceInt.cols());
		auto tmp = new Matrix<float, 84, 84>(matrixData); //Use of heap due to size
		m_TraceInt = *tmp;
		delete tmp;

		input.read((char*)matrixData, sizeof(float) * m_TraceOut.rows() * m_TraceOut.cols());
		m_TraceOut = Matrix<float, 84, 1>(matrixData);
	}

	delete[] buffer;
	delete[] matrixData;
	input.close();
}

AILearning AI::GetAILearning() const
{
	return m_AILearning;
}

int AI::PlayMove(Grid* pGrid, bool asPlayer1, bool trainingMode)
{
	int playedMove{}, bestMove{};

	//Find the best move acording to the NN
	vector<float> probabilities;
	probabilities.reserve(7);
	Matrix<float, 1, 84> gridStateSavePre{ pGrid->GetStateMatrix() }; //gridState always puts P1 in the first half and P2 in the second half of the matrix
	Matrix<float, 1, 84> gridStateSavePost{ pGrid->GetStateMatrix() };

	for (int i{}; i < 7; ++i)
	{
		int row{ pGrid->GetAvailableRowInColumn(i) };
		if (row == -1)
			probabilities.push_back(asPlayer1 ? -FLT_MAX : FLT_MAX);
		else
		{
			gridStateSavePost(0, row * 7 + i + (asPlayer1 ? 0 : 42)) = 1.0f;
			probabilities.push_back(NNForwardPass(gridStateSavePost));
			gridStateSavePost(0, row * 7 + i + (asPlayer1 ? 0 : 42)) = 0.0f;
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
		//E-Greedy test. If value is below epsilon, play random move.
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
	output.open(filename, std::ios::binary);

	int AILearningMethod{ int(m_AILearning) };
	output.write((const char*)&AILearningMethod, sizeof(int));
	output.write((const char*)&m_Epsilon, sizeof(float));
	output.write((const char*)&m_LearningRate, sizeof(float));
	if (m_AILearning == AILearning::TDLambda) output.write((const char*)&m_Lambda, sizeof(float));
	output.write((const char*)m_Wint.data(), sizeof(float) * (uint32_t)m_Wint.rows() * (uint32_t)m_Wint.cols());
	output.write((const char*)m_Wout.data(), sizeof(float) * (uint32_t)m_Wout.rows() * (uint32_t)m_Wout.cols());
	if (m_AILearning == AILearning::TDLambda)output.write((const char*)m_TraceInt.data(), sizeof(float) * (uint32_t)m_TraceInt.rows() * (uint32_t)m_TraceInt.cols());
	if (m_AILearning == AILearning::TDLambda)output.write((const char*)m_TraceOut.data(), sizeof(float) * (uint32_t)m_TraceOut.rows() * (uint32_t)m_TraceOut.cols());

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
	Matrix<float, 1, s_InnerLayerNeuronCount> intermediate{ oldState * m_Wint };
	Matrix<float, 1, s_InnerLayerNeuronCount> oldStatePInt = Sigmoid(intermediate);
	float oldStatePOut{ NNForwardPass(oldState) };
	float delta{ oldStatePOut - NNForwardPass(bestState) };
	float gradOut{ oldStatePOut * (1 - oldStatePOut) };

	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		float gradInt = oldStatePInt(0, i) * (1 - oldStatePInt(0, i));
		float deltaInt = gradOut * m_Wout(i, 0) * gradInt;
		for (int j{}; j < 84; ++j)
		{
			m_Wint(j, i) -= m_LearningRate * delta * deltaInt * oldState(0, j);
		}
		m_Wout(i, 0) -= m_LearningRate * delta * gradOut * oldStatePInt(0, i);
	}
}

void AI::NNQLearningFinal(const Eigen::Matrix<float, 1, 84>& oldState, float result)
{
	Matrix<float, 1, s_InnerLayerNeuronCount> intermediate{ oldState * m_Wint };
	Matrix<float, 1, s_InnerLayerNeuronCount> oldStatePInt = Sigmoid(intermediate);

	float delta{ NNForwardPass(oldState) - result };

	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		for (int j{}; j < 84; ++j)
		{
			m_Wint(j, i) -= m_LearningRate * delta * result * oldState(0, j);
		}
		m_Wout(i, 0) -= m_LearningRate * delta * result * oldStatePInt(0, i);
	}
}

void AI::NNTDLambda(const Matrix<float, 1, 84>& oldState, const Matrix<float, 1, 84>& playedState)
{
}

float AI::Sigmoid(float x) const
{
	return 1.0f / (1.0f + exp(-x));
}

const Matrix<float, 1, AI::s_InnerLayerNeuronCount>& AI::Sigmoid(Matrix<float, 1, s_InnerLayerNeuronCount>& m) const
{
	for (int i{}; i < s_InnerLayerNeuronCount; ++i)
	{
		m(0, i) = Sigmoid(m(0, i));
	}
	return m;
}
