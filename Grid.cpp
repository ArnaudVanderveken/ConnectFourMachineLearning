#include "pch.h"
#include "Grid.h"


Grid::Grid()
{
	m_Grid = new char* [s_NrRows];
	for (int i{}; i < Grid::s_NrRows; ++i)
	{
		m_Grid[i] = new char[s_NrColumns];
	}

	ResetGrid();
}

Grid::~Grid()
{
	for (int i{}; i < Grid::s_NrRows; ++i)
	{
		delete[] m_Grid[i];
	}
	delete[] m_Grid;
}

void Grid::ResetGrid()
{
	for (int i{}; i < Grid::s_NrRows; ++i)
	{
		for (int j{}; j < Grid::s_NrColumns; ++j)
		{
			m_Grid[i][j] = EMPTY_TOKEN;
		}
	}
	m_StateMatrix.fill(0.0f);
}

void Grid::InsertToken(char token, int column)
{
	int i{ Grid::s_NrRows - 1 };
	do 
	{
		if (m_Grid[i][column] == EMPTY_TOKEN)
		{
			m_Grid[i][column] = token;

			if (token == P1_TOKEN)
			{
				m_StateMatrix(0, i * s_NrColumns + column) = token == P1_TOKEN ? 1.0f : -1.0f;
			}
			return;
		}
		--i;
	} while (i >= 0);
}

int Grid::GetAvailableRowInColumn(int column) const
{
	int i{ Grid::s_NrRows - 1 };
	do
	{
		if (m_Grid[i][column] == EMPTY_TOKEN)
		{
			return i;
		}
		--i;
	} while (i >= 0);
	return -1;
}

bool Grid::IsColumnFull(int column) const
{
	return m_Grid[0][column] != EMPTY_TOKEN;
}

char** Grid::GetGrid() const
{
	return m_Grid;
}

void Grid::Print() const
{
	for (int i{}; i < Grid::s_NrRows; ++i)
	{
		std::cout << "|";
		for (int j{}; j < Grid::s_NrColumns; ++j)
		{
			std::cout << m_Grid[i][j];
		}
		std::cout << "|\n";
	}
	std::cout << std::endl;
}

const Eigen::Matrix<float, 1, 42>& Grid::GetStateMatrix() const
{
	return m_StateMatrix;
}

Grid::WinState Grid::CheckWinCondition() const
{
	//Check Player Win
	for (int i{ Grid::s_NrRows-1 }; i >= 0; --i)
	{
		for (int j{}; j < (Grid::s_NrColumns + 1) / 2; ++j) //Only need ot check columns 0 1 2 and 3
		{
			char token = m_Grid[i][j];

			//Skip empty cells
			if (token == EMPTY_TOKEN) continue;

			//Top Half:
			// - Horizontal check
			// - Diagonal Down-Right check
			if (i < Grid::s_NrRows / 2)
			{
				if ((m_Grid[i][j + 1] == token && m_Grid[i][j + 2] == token && m_Grid[i][j + 3] == token)
					|| (m_Grid[i + 1][j + 1] == token && m_Grid[i + 2][j + 2] == token && m_Grid[i + 3][j + 3] == token))
				{
					if (token == P1_TOKEN)
						return WinState::p1;
					else
						return WinState::p2;
				}
			}

			//Bottom Half:
			// - Horizontal check
			// - Vertical check
			// - Diagonal Up-Right check
			else
			{
				if ((m_Grid[i][j + 1] == token && m_Grid[i][j + 2] == token && m_Grid[i][j + 3] == token)
					|| (m_Grid[i - 1][j] == token && m_Grid[i - 2][j] == token && m_Grid[i - 3][j] == token)
					|| (m_Grid[i - 1][j + 1] == token && m_Grid[i - 2][j + 2] == token && m_Grid[i - 3][j + 3] == token))
				{
					if (token == P1_TOKEN)
						return WinState::p1;
					else
						return WinState::p2;
				}
			}
		}
	}

	//Draw Check
	bool draw{ true };
	for (int j{}; j < Grid::s_NrColumns; ++j)
	{
		draw = draw && IsColumnFull(j);
	}
	if (draw) return WinState::draw;

	//No Winner
	return WinState::none;
}
