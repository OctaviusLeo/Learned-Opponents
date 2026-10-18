// One game mode for both maps: RaceLevel spawns the race, TransferArena spawns the headless
// transfer harness. The maps themselves are empty; everything is built from code at BeginPlay.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "LOGameMode.generated.h"

UCLASS()
class LEARNEDOPPONENTS_API ALOGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	ALOGameMode();

protected:
	virtual void BeginPlay() override;
};
