#include "Race/LOGameMode.h"

#include "Engine/World.h"
#include "LearnedOpponents.h"
#include "Race/LOHUD.h"
#include "Race/LORaceManager.h"
#include "Transfer/LOTransferHarness.h"

ALOGameMode::ALOGameMode()
{
	DefaultPawnClass = nullptr;
	HUDClass = ALOHUD::StaticClass();
}

void ALOGameMode::BeginPlay()
{
	Super::BeginPlay();
	const FString Map = GetWorld()->GetMapName();
	if (Map.Contains(TEXT("TransferArena")))
	{
		UE_LOG(LogLearnedOpponents, Display, TEXT("TransferArena: starting the transfer harness"));
		GetWorld()->SpawnActor<ALOTransferHarness>();
	}
	else
	{
		UE_LOG(LogLearnedOpponents, Display, TEXT("%s: starting the race"), *Map);
		GetWorld()->SpawnActor<ALORaceManager>();
	}
}
