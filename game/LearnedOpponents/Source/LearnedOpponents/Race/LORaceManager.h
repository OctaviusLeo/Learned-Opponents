// The race: the player drives one robot (W/S forward and back, A/D turn), the ONNX policy drives
// the other, both racing through the same checkpoints between obstacles. Both robots use the
// same kinematic differential drive and step at control_dt_s on a fixed-step accumulator.
//
// Command line: -LOModel=<path.onnx> (default results/models/randomized_seed0.onnx),
// -LOCourse=<path.json> (default results/game/race_course.json), -LODemoPlayer (a scripted
// controller drives the player robot, for the README GIF), -LOQuitAfter=<seconds>.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Inference/LOPolicy.h"
#include "Obs/LOGameData.h"
#include "LORaceManager.generated.h"

class ALOArena;
class ALORobot;
class AStaticMeshActor;

struct FLORacer
{
	TObjectPtr<ALORobot> Robot;
	int32 NextCheckpoint = 0;
	int32 Laps = 0;
	int32 Crashes = 0;
};

UCLASS()
class LEARNEDOPPONENTS_API ALORaceManager : public AActor
{
	GENERATED_BODY()

public:
	ALORaceManager();
	virtual void Tick(float DeltaSeconds) override;

	// HUD text lines.
	TArray<FString> StatusLines() const;

protected:
	virtual void BeginPlay() override;

private:
	void ControlStep();
	void Advance(FLORacer& Racer, bool bCollided);
	void PlayerAction(double& OutLeft, double& OutRight) const;
	void DemoAction(double& OutLeft, double& OutRight) const;
	void Respawn(FLORacer& Racer);
	FVector2D Checkpoint(int32 Index) const { return Checkpoints[Index % Checkpoints.Num()]; }

	UPROPERTY()
	TObjectPtr<ALOArena> Arena;
	UPROPERTY()
	TArray<TObjectPtr<AActor>> Markers;

	FLORacer Opponent;
	FLORacer Player;
	FLOTaskData Task;
	FLOPolicy Policy;
	TArray<FVector2D> Checkpoints;
	FVector StartOpponent;   // spec x, y, yaw
	FVector StartPlayer;
	FString ModelName;
	FString Error;
	double Accumulator = 0.0;
	double RaceTime = 0.0;
	double QuitAfter = -1.0;
	bool bDemoPlayer = false;
	bool bReady = false;
};
