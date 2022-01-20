#pragma once


class AI final
{
public:
	AI(float epsilon, float learningRate, float lambda);
	~AI() = default;
	AI(const AI&) = delete;
	AI& operator=(const AI&) = delete;
	AI(AI&&) noexcept = delete;
	AI& operator=(AI&&) noexcept = delete;

private:
	static const int s_InnerLayerNeuronCount{ 84 };
	const float m_Epsilon;
	const float m_LearningRate;
	const float m_Lambda;

	Eigen::Matrix<float, 84, s_InnerLayerNeuronCount> m_Wint;
	Eigen::Matrix<float, s_InnerLayerNeuronCount, 1> m_Wout;

	float NNForwardPass(Eigen::Matrix<float, 1, 84> input) const;
	void NNQLearning(Eigen::Matrix<float, 1, 84> oldState, Eigen::Matrix<float, 1, 84> bestState);
	void NNTDLambda(Eigen::Matrix<float, 1, 84> oldState, Eigen::Matrix<float, 1, 84> playedState);
	float Sigmoid(float x) const;
	const Eigen::Matrix<float, 1, s_InnerLayerNeuronCount>& Sigmoid(Eigen::Matrix<float, 1, s_InnerLayerNeuronCount>& m) const;
};

