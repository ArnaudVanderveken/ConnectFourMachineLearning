#include "pch.h"
#include "Grid.h"
#include "AI.h"

using std::cout, std::cin, std::endl, std::string;

void Train(Grid* pGrid, AI* pAI, uint32_t rounds)
{

}

void Play(Grid* pGrid, AI* pAI, bool isAIP1)
{
	bool gameLoop{ true };
	uint32_t roundCounter{};

	int aiPlay{}, playerPlay{};
	Grid::WinState winState;

	while (gameLoop)
	{
		++roundCounter;
		if (isAIP1)
		{
			aiPlay = pAI->PlayMove(pGrid, isAIP1, false);
			pGrid->InsertToken(P1_TOKEN, aiPlay);
			cout << "AI played in column " << aiPlay << endl;
			pGrid->Print();

			// Ask player input
			bool validInput{};
			do {
				cout << "Column to play in ?" << endl;
				cin >> playerPlay;
				if (playerPlay >= 0 && playerPlay <= 6 && !pGrid->IsColumnFull(playerPlay))
					validInput = true;
			} while (!validInput);
			pGrid->InsertToken(P2_TOKEN, playerPlay);
			pGrid->Print();

			winState = pGrid->CheckWinCondition();
			switch (winState)
			{
			case Grid::WinState::none:
				break;

			case Grid::WinState::p1:
				cout << "AI win\n" << endl;
				gameLoop = false;
				break;

			case Grid::WinState::p2:
				cout << "You win\n" << endl;
				gameLoop = false;
				break;

			case Grid::WinState::draw:
				cout << "Draw\n" << endl;
				gameLoop = false;
				break;
			}
		}
		else
		{
			// Ask player input
			bool validInput{};
			do {
				cout << "Column to play in ?" << endl;
				cin >> playerPlay;
				if (playerPlay >= 0 && playerPlay <= 6 && !pGrid->IsColumnFull(playerPlay))
					validInput = true;
			} while (!validInput);
			pGrid->InsertToken(P1_TOKEN, playerPlay);
			pGrid->Print();

			aiPlay = pAI->PlayMove(pGrid, isAIP1, false);
			pGrid->InsertToken(P2_TOKEN, aiPlay);
			cout << "AI played in column " << aiPlay << endl;
			pGrid->Print();

			winState = pGrid->CheckWinCondition();
			switch (winState)
			{
			case Grid::WinState::none:
				break;

			case Grid::WinState::p1:
				cout << "You win\n" << endl;
				gameLoop = false;
				break;

			case Grid::WinState::p2:
				cout << "AI win\n" << endl;
				gameLoop = false;
				break;

			case Grid::WinState::draw:
				cout << "Draw\n" << endl;
				gameLoop = false;
				break;
			}
		}
	}
}

int main()
{
	auto pGrid{ std::make_unique<Grid>() };

	GameMode gameMode{};
	AILearning aiLearning{};
	float epsilon{}, learningRate{}, lambda{};

	bool gameLoop{ true };

	string command{};

	// Select Gamemode
	do {
		cout << "GameMode: (play - train)" << endl;
		cin >> command;
	} while (command != "play" && command != "train");

	if (command == "play")
		gameMode = GameMode::play;
	else
		gameMode = GameMode::train;


	//Select AI learning method
	do {
		cout << "AILearning: (QLearning - TDLambda)" << endl;
		cin >> command;
	} while (command != "QLearning" && command != "TDLambda");

	if (command == "QLearning")
		aiLearning = AILearning::QLearning;
	else
		aiLearning = AILearning::TDLambda;

	// Set epsilon
	cout << "Epsilon value: (float clamped between 0.0f and 1.0f)" << endl;
	cin >> epsilon;

	// Set LearningRate
	cout << "Learning rate: (float clamped between 0.0f and 1.0f)" << endl;
	cin >> learningRate;

	if (aiLearning == AILearning::TDLambda)
	{
		//Set Lambda
		cout << "Lambda value: (float clamped between 0.0f and 1.0f)" << endl;
		cin >> lambda;
	}

	auto pAI{ std::make_unique<AI>(epsilon, learningRate, lambda) };

	uint32_t roundsCounter{};

	switch (gameMode)
	{
	case GameMode::play:
		while (gameLoop)
		{
			++roundsCounter;
			Play(pGrid.get(), pAI.get(), (roundsCounter % 2) == 0);

			// ASk for rematch
			do {
				cout << "Play again ? [y/n] ";
				cin >> command;
			} while (command != "y" && command != "n");

			if (command == "y")
			{
				pGrid->ResetGrid();
				cout << endl;
			}
			else
			{
				gameLoop = false;
			}
		}
		break;

	case GameMode::train:
		uint32_t rounds{};
		// Set number of rounds;
		cout << "Nr of training rounds: (uint32_t)" << endl;
		cin >> rounds;

		break;
	}
}