#pragma once

using std::string, std::ofstream;

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

	AILearning GetAILearning() const;

	int PlayMove(Grid* pGrid, bool asPlayer1, bool trainingMode);

	void SaveToFile(string filename);

private:
	static const int s_InnerLayerNeuronCount{ 84 };
	AILearning m_AILearning;
	float m_Epsilon;
	float m_LearningRate;
	float m_Lambda;

	Eigen::Matrix<float, 84, s_InnerLayerNeuronCount> m_Wint;
	Eigen::Matrix<float, s_InnerLayerNeuronCount, 1> m_Wout;

	float NNForwardPass(const Eigen::Matrix<float, 1, 84>& input) const;
	void NNQLearning(const Eigen::Matrix<float, 1, 84>& oldState, const Eigen::Matrix<float, 1, 84>& bestState);
	void NNTDLambda(const Eigen::Matrix<float, 1, 84>& oldState, const Eigen::Matrix<float, 1, 84>& playedState);
	float Sigmoid(float x) const;
	const Eigen::Matrix<float, 1, s_InnerLayerNeuronCount>& Sigmoid(Eigen::Matrix<float, 1, s_InnerLayerNeuronCount>& m) const;
};

