#include "pch.h"
#include "Grid.h"
#include "AI.h"


using std::cout, std::cin, std::endl, std::string, std::ifstream, std::regex, std::regex_match, Eigen::Matrix;


void Train(Grid* pGrid, AI* pAI, uint32_t rounds)
{
	for (uint32_t gamesCounter{}; gamesCounter < rounds; ++gamesCounter)
	{
		bool gameLoop{ true };
		uint32_t roundsCounter{};
		int aiPlay{};

		Matrix<float, 1, 84> gridStateSave;

		while (gameLoop)
		{
			++roundsCounter;

			// AI as P1
			aiPlay = pAI->PlayMove(pGrid, (gamesCounter % 2) == 0, roundsCounter != 1u);
			pGrid->InsertToken(P1_TOKEN, aiPlay);

			//Check Win conditions
			switch (pGrid->CheckWinCondition())
			{
			case Grid::WinState::p1:
				gridStateSave = pGrid->GetStateMatrix();
				pAI->NNQLearningFinal(gridStateSave, WIN_CREDITS);
				gameLoop = false;
				break;

			case Grid::WinState::p2:
				gridStateSave = pGrid->GetStateMatrix();
				pAI->NNQLearningFinal(gridStateSave, LOSS_CREDITS);
				gameLoop = false;
				break;

			case Grid::WinState::draw:
				gridStateSave = pGrid->GetStateMatrix();
				pAI->NNQLearningFinal(gridStateSave, DRAW_CREDITS);
				gameLoop = false;
				break;

			default:
				break;
			}

			//Early exit test
			if (!gameLoop) break;

			//AI as P2
			aiPlay = pAI->PlayMove(pGrid, (gamesCounter % 2) == 1, roundsCounter != 1u);
			pGrid->InsertToken(P2_TOKEN, aiPlay);

			//Check Win conditions
			switch (pGrid->CheckWinCondition())
			{
			case Grid::WinState::p1:
				gridStateSave = pGrid->GetStateMatrix();
				pAI->NNQLearningFinal(gridStateSave, WIN_CREDITS);
				gameLoop = false;
				break;

			case Grid::WinState::p2:
				gridStateSave = pGrid->GetStateMatrix();
				pAI->NNQLearningFinal(gridStateSave, LOSS_CREDITS);
				gameLoop = false;
				break;

			case Grid::WinState::draw:
				gridStateSave = pGrid->GetStateMatrix();
				pAI->NNQLearningFinal(gridStateSave, DRAW_CREDITS);
				gameLoop = false;
				break;

			default:
				break;
			}
		}
		pGrid->ResetGrid();
	}
}

void Play(Grid* pGrid, AI* pAI, bool isAIP1)
{
	bool gameLoop{ true };

	int aiPlay{}, playerPlay{};

	while (gameLoop)
	{
		if (isAIP1)
		{
			//AI move
			aiPlay = pAI->PlayMove(pGrid, isAIP1, false);
			pGrid->InsertToken(P1_TOKEN, aiPlay);
			cout << "AI played in column " << aiPlay << endl;
			pGrid->Print();

			//Check win conditions
			switch (pGrid->CheckWinCondition())
			{
			case Grid::WinState::none:
				break;

			case Grid::WinState::p1:
				cout << "AI win\n" << endl;
				gameLoop = false;
				return;

			case Grid::WinState::p2:
				cout << "You win\n" << endl;
				gameLoop = false;
				return;

			case Grid::WinState::draw:
				cout << "Draw\n" << endl;
				gameLoop = false;
				return;
			}

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

			//Check win conditions
			switch (pGrid->CheckWinCondition())
			{
			case Grid::WinState::none:
				break;

			case Grid::WinState::p1:
				cout << "AI win\n" << endl;
				gameLoop = false;
				return;

			case Grid::WinState::p2:
				cout << "You win\n" << endl;
				gameLoop = false;
				return;

			case Grid::WinState::draw:
				cout << "Draw\n" << endl;
				gameLoop = false;
				return;
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

			//Check win conditions
			switch (pGrid->CheckWinCondition())
			{
			case Grid::WinState::none:
				break;

			case Grid::WinState::p1:
				cout << "You win\n" << endl;
				gameLoop = false;
				return;

			case Grid::WinState::p2:
				cout << "AI win\n" << endl;
				gameLoop = false;
				return;

			case Grid::WinState::draw:
				cout << "Draw\n" << endl;
				gameLoop = false;
				return;
			}

			//AI move
			aiPlay = pAI->PlayMove(pGrid, isAIP1, false);
			pGrid->InsertToken(P2_TOKEN, aiPlay);
			cout << "AI played in column " << aiPlay << endl;
			pGrid->Print();

			//Check Win conditions
			switch (pGrid->CheckWinCondition())
			{
			case Grid::WinState::none:
				break;

			case Grid::WinState::p1:
				cout << "You win\n" << endl;
				gameLoop = false;
				return;

			case Grid::WinState::p2:
				cout << "AI win\n" << endl;
				gameLoop = false;
				return;

			case Grid::WinState::draw:
				cout << "Draw\n" << endl;
				gameLoop = false;
				return;
			}
		}
	}
}

AI* SetUpAI()
{
	AILearning aiLearning{};
	float epsilon{}, learningRate{}, lambda{};

	string command;

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

	return new AI(aiLearning, epsilon, learningRate, lambda);
}

AI* LoadAI(string filename)
{
	return new AI(filename);
}

void SaveAI(AI* pAI)
{
	string filename;
	cout << "Enter filename (without extension): " << endl;
	cin >> filename;
	filename += ".aidata";

	pAI->SaveToFile(filename);
}

int main()
{
	//Random seed
	srand(unsigned(time(0)));

	auto pGrid{ std::make_unique<Grid>() };
	AI* pAI{};
	GameMode gameMode{};

	string command{};
	bool mainLoop{ true }, gameLoop{ true };

	while (mainLoop)
	{

		// Select Gamemode
		do {
			cout << "GameMode: (play - train - save - quit)" << endl;
			cin >> command;
			if (command == "save")
			{
				if (!pAI)
				{
					cout << "No AI to save." << endl;
					continue;
				}
				SaveAI(pAI);
			}
		} while (command != "play" && command != "train" && command != "quit");

		//Exit program
		if (command == "quit")
		{
			mainLoop = false;
			break;
		}

		if (command == "play")
			gameMode = GameMode::play;
		else
			gameMode = GameMode::train;

		// Selet AI
		bool valid{};
		do {
			cout << "AI Setup: (create - load - keep)" << endl;
			cin >> command;

			valid = (command == "create") || (command == "load") || (command == "keep");

			if (command == "keep" && pAI == nullptr)
			{
				cout << "No AI in memory" << endl;
				valid = false;
			}
		} while (!valid);


		if (command == "create")
		{
			//Delete old AI if existing
			if (pAI != nullptr) delete pAI;
			pAI = SetUpAI();
		}
		else if (command == "load")
		{
			//Delete old AI if existing
			if (pAI != nullptr) delete pAI;

			bool valid{};
			do {
				cout << "File to load from: (with .aidata extension)" << endl;
				cin >> command;

				const regex extension{ ".+\\.aidata$" };

				valid = regex_match(command, extension);

			} while (!valid);

			pAI = LoadAI(command);
		}

		uint32_t roundsCounter{};

		switch (gameMode)
		{
		case GameMode::play:
			while (gameLoop)
			{
				++roundsCounter;
				Play(pGrid.get(), pAI, (roundsCounter % 2) == 0);

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

			Train(pGrid.get(), pAI, rounds);

			break;
		}
	}

	//CleanUP
	delete pAI;

	return 0;
}