// Task configuration and layouts shared with Python, read from JSON files under the repository's
// results/ directory (results/game/task.json, results/transfer/episodes.json, results/game/race_course.json).

#pragma once

#include "CoreMinimal.h"
#include "Obs/LOObsSpec.h"

class FJsonObject;

enum class ELOObstacleKind : uint8
{
	Inactive = 0,
	Box = 1,
	Cylinder = 2,
};

struct LEARNEDOPPONENTS_API FLOObstacle
{
	ELOObstacleKind Kind = ELOObstacleKind::Inactive;
	double X = 0.0;      // spec frame, metres
	double Y = 0.0;
	double Size = 0.0;   // box side or cylinder diameter, metres
};

struct LEARNEDOPPONENTS_API FLOLayout
{
	TArray<FLOObstacle> Obstacles;
	double StartX = 0.0;
	double StartY = 0.0;
	double StartYaw = 0.0;
	double GoalX = 0.0;
	double GoalY = 0.0;
};

struct LEARNEDOPPONENTS_API FLOTaskData
{
	FLOSpecConfig Spec;
	double RobotRadiusM = 0.15;
	double RayHeightM = 0.15;
	double ControlDtS = 0.05;
	double PhysicsDtS = 0.01;
	double ArrivalToleranceM = 0.25;
	double WallHeightM = 0.5;
	double ObstacleHeightM = 0.5;
	int32 EpisodeLimit = 400;
	int32 MaxObstacles = 8;
	FString Commit;

	int32 Substeps() const { return FMath::Max(1, FMath::RoundToInt(ControlDtS / PhysicsDtS)); }
};

namespace LOData
{
	// Repository root (two levels above the .uproject directory) and paths under results/.
	LEARNEDOPPONENTS_API FString RepoRoot();
	LEARNEDOPPONENTS_API FString ResultsPath(const FString& Relative);

	LEARNEDOPPONENTS_API TSharedPtr<FJsonObject> LoadJson(const FString& Path, FString& OutError);
	LEARNEDOPPONENTS_API bool ParseSpec(const TSharedPtr<FJsonObject>& Obj, FLOSpecConfig& Out);
	LEARNEDOPPONENTS_API bool LoadTask(const FString& Path, FLOTaskData& Out, FString& OutError);
	LEARNEDOPPONENTS_API bool ParseLayout(const TSharedPtr<FJsonObject>& Episode, FLOLayout& Out);
	LEARNEDOPPONENTS_API bool LoadLayouts(const FString& Path, TArray<FLOLayout>& Out, FString& OutError);

	// Command-line value "-Key=Value" or Default.
	LEARNEDOPPONENTS_API FString CmdValue(const TCHAR* Key, const FString& Default);
}
