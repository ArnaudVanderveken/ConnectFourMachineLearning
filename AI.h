#pragma once


class AI final
{
public:
	AI() = default;
	~AI() = default;
	AI(const AI&) = delete;
	AI& operator=(const AI&) = delete;
	AI(AI&&) noexcept = delete;
	AI& operator=(AI&&) noexcept = delete;

private:
	static const int s_InnerLayerNeuronCount{ 84 };

	Eigen::Matrix<float, 84, s_InnerLayerNeuronCount> m_Win;
	Eigen::Matrix<float, s_InnerLayerNeuronCount, 1> m_Wout;
};

