#pragma once

using std::string;

class Grid;

class AI final
{
public:
	AI(AILearning aiLearning, float epsilon, float learningRate, float lambda);
	AI(string filename);
	~AI() = default;
	AI(const AI&) = delete;
	AI& operator=(const AI&) = delete;
	AI(AI&&) noexcept = delete;
	AI& operator=(AI&&) noexcept = delete;

	int PlayMove(Grid* pGrid, bool asPlayer1, bool trainingMode);
	void NNQLearningFinal(const Eigen::Matrix<float, 1, 42>& oldState, float result);

	void SaveToFile(string filename);

private:
	AILearning m_AILearning;
	float m_Epsilon;
	float m_LearningRate;
	float m_Lambda;

	Eigen::Matrix<float, 42, 1> m_Weights, m_Trace;

	float NNForwardPass(const Eigen::Matrix<float, 1, 42>& input) const;
	void NNQLearning(const Eigen::Matrix<float, 1, 42>& oldState, const Eigen::Matrix<float, 1, 42>& bestState);
	void NNTDLambda(const Eigen::Matrix<float, 1, 42>& oldState, const Eigen::Matrix<float, 1, 42>& playedState);
	float Sigmoid(float x) const;
	float ReLU(float x) const;
	float SWISH(float x) const;
};

