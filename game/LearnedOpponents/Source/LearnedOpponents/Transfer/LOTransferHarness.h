// Headless transfer harness for /Game/Maps/TransferArena (task 17.1).
//
// Runs every exported policy on every layout of results/transfer/episodes.json with the same
// termination rules as MuJoCo (collision > success > timeout, checked after each control step).
// Each episode is stepped manually at control_dt_s inside a single frame (trace rays, build the
// observation, run NNE inference, apply the swept kinematic move), so the result does not depend
// on the frame rate. Two frames pass after each layout change so scene queries see the moved
// obstacles. Results go to results/transfer/game_results.json, then the game exits.
//
// Command line: -LOEpisodes=<json> -LOModels=<name,name,...> -LOOut=<json>.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Inference/LOPolicy.h"
#include "Obs/LOGameData.h"
#include "LOTransferHarness.generated.h"

class ALOArena;
class ALORobot;

UCLASS()
class LEARNEDOPPONENTS_API ALOTransferHarness : public AActor
{
	GENERATED_BODY()

public:
	ALOTransferHarness();
	virtual void Tick(float DeltaSeconds) override;

	// Play one episode synchronously; returns "success", "collision", or "timeout".
	static FString RunEpisode(ALORobot* Robot, FLOPolicy& Policy, const FLOLayout& Layout, int32& OutSteps);

protected:
	virtual void BeginPlay() override;

private:
	void Finish(bool bOk);

	UPROPERTY()
	TObjectPtr<ALOArena> Arena;
	UPROPERTY()
	TObjectPtr<ALORobot> Robot;

	FLOTaskData Task;
	TArray<FLOLayout> Layouts;
	TArray<FString> ModelNames;
	TArray<FString> Reasons;     // per (model, episode), in order
	TArray<int32> Steps;
	FLOPolicy Policy;
	FString OutPath;
	FString EpisodesPath;
	int32 ModelIndex = 0;
	int32 EpisodeIndex = 0;
	int32 WaitFrames = 0;
	bool bLayoutApplied = false;
	bool bDone = false;
	double StartSeconds = 0.0;
};
