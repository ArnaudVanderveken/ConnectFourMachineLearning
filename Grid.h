#pragma once


class Grid final
{
public:
	enum class WinState
	{
		none, p1, p2, draw
	};

	Grid();
	~Grid();
	Grid(const Grid&) = delete;
	Grid& operator=(const Grid&) = delete;
	Grid(Grid&&) noexcept = delete;
	Grid& operator=(Grid&&) noexcept = delete;

	static const int s_NrRows{ 6 };
	static const int s_NrColumns{ 7 };

	void ResetGrid();
	void InsertToken(char token, int column);
	int GetAvailableRowInColumn(int column) const;
	bool IsColumnFull(int column) const;

	char** GetGrid() const;
	void Print() const;

	const Eigen::Matrix<float, 1, 84>& GetStateMatrix() const;

	WinState CheckWinCondition() const;

private:
	char** m_Grid;
	Eigen::Matrix<float, 1, 84> m_StateMatrix;

};

