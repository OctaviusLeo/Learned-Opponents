#include "Race/LORaceManager.h"

#include "Camera/CameraActor.h"
#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Components/LightComponent.h"
#include "Engine/DirectionalLight.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "HAL/PlatformMisc.h"
#include "InputCoreTypes.h"
#include "LearnedOpponents.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Obs/LOObsSpec.h"
#include "Race/LOArena.h"
#include "Race/LORobot.h"

ALORaceManager::ALORaceManager()
{
	PrimaryActorTick.bCanEverTick = true;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

static bool ReadPose(const TSharedPtr<FJsonObject>& Obj, const TCHAR* Field, FVector& Out)
{
	const TSharedPtr<FJsonObject>* P = nullptr;
	return Obj->TryGetObjectField(Field, P) && (*P)->TryGetNumberField(TEXT("x"), Out.X) &&
		(*P)->TryGetNumberField(TEXT("y"), Out.Y) && (*P)->TryGetNumberField(TEXT("yaw"), Out.Z);
}

void ALORaceManager::BeginPlay()
{
	Super::BeginPlay();
	FString Err;
	if (!LOData::LoadTask(LOData::ResultsPath(TEXT("game/task.json")), Task, Err))
	{
		Error = Err;
		UE_LOG(LogLearnedOpponents, Error, TEXT("%s"), *Err);
		return;
	}
	const FString CoursePath = LOData::CmdValue(TEXT("LOCourse"), LOData::ResultsPath(TEXT("game/race_course.json")));
	const TSharedPtr<FJsonObject> Course = LOData::LoadJson(CoursePath, Err);
	FLOLayout Layout;
	const TArray<TSharedPtr<FJsonValue>>* Points = nullptr;
	if (!Course.IsValid() || !LOData::ParseLayout(Course, Layout) || !Course->TryGetArrayField(TEXT("checkpoints"), Points) ||
		!ReadPose(Course, TEXT("start"), StartOpponent) || !ReadPose(Course, TEXT("player_start"), StartPlayer))
	{
		Error = Err.IsEmpty() ? FString::Printf(TEXT("%s is malformed"), *CoursePath) : Err;
		UE_LOG(LogLearnedOpponents, Error, TEXT("%s"), *Error);
		return;
	}
	for (const TSharedPtr<FJsonValue>& V : *Points)
	{
		const TSharedPtr<FJsonObject> P = V->AsObject();
		Checkpoints.Add(FVector2D(P->GetNumberField(TEXT("x")), P->GetNumberField(TEXT("y"))));
	}

	const FString ModelPath = LOData::CmdValue(TEXT("LOModel"), LOData::ResultsPath(TEXT("models/randomized_seed0.onnx")));
	ModelName = FPaths::GetBaseFilename(ModelPath);
	if (!Policy.Load(ModelPath, Task.Spec.ObsDim(), Err))
	{
		Error = Err;
		UE_LOG(LogLearnedOpponents, Error, TEXT("policy: %s"), *Err);
		return;
	}
	bDemoPlayer = FParse::Param(FCommandLine::Get(), TEXT("LODemoPlayer"));
	const FString Quit = LOData::CmdValue(TEXT("LOQuitAfter"), TEXT(""));
	QuitAfter = Quit.IsEmpty() ? -1.0 : FCString::Atod(*Quit);

	UWorld* World = GetWorld();
	Arena = World->SpawnActor<ALOArena>();
	Arena->Build(Task);
	Arena->ApplyLayout(Layout);
	Opponent.Robot = World->SpawnActor<ALORobot>();
	Opponent.Robot->Configure(Task, FLinearColor(0.10f, 0.35f, 0.90f));
	Player.Robot = World->SpawnActor<ALORobot>();
	Player.Robot->Configure(Task, FLinearColor(0.10f, 0.75f, 0.30f));
	Opponent.Robot->SetPose(StartOpponent.X, StartOpponent.Y, StartOpponent.Z);
	Player.Robot->SetPose(StartPlayer.X, StartPlayer.Y, StartPlayer.Z);

	UStaticMesh* Disc = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
	for (const FVector2D& C : Checkpoints)
	{
		AStaticMeshActor* M = World->SpawnActor<AStaticMeshActor>();
		M->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);
		M->GetStaticMeshComponent()->SetStaticMesh(Disc);
		M->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		M->GetStaticMeshComponent()->SetMaterial(0, ALOArena::ColorMaterial(M, FLinearColor(0.95f, 0.80f, 0.10f)));
		M->SetActorLocation(LOObs::SpecToUnrealLocation(C.X, C.Y, 1.0));
		const double D = 2.0 * Task.ArrivalToleranceM;
		M->SetActorScale3D(FVector(D, D, 0.02));
		Markers.Add(M);
	}

	ADirectionalLight* Light = World->SpawnActor<ADirectionalLight>(FVector(0, 0, 1000), FRotator(-60.0, 35.0, 0.0));
	if (Light)
	{
		Light->GetLightComponent()->SetMobility(EComponentMobility::Movable);
	}
	ACameraActor* Camera = World->SpawnActor<ACameraActor>(FVector(0, 0, 1150), FRotator(-90.0, 0.0, 0.0));
	if (APlayerController* PC = World->GetFirstPlayerController(); PC && Camera)
	{
		PC->SetViewTarget(Camera);
	}
	bReady = true;
	UE_LOG(LogLearnedOpponents, Display, TEXT("race ready: model %s via %s, %d checkpoints, demo player %d"),
		*ModelPath, FLOPolicy::RuntimeName, Checkpoints.Num(), bDemoPlayer ? 1 : 0);
}

void ALORaceManager::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (!bReady)
	{
		return;
	}
	RaceTime += DeltaSeconds;
	Accumulator += DeltaSeconds;
	int32 Steps = 0;
	while (Accumulator >= Task.ControlDtS && Steps < 10)   // fixed step; cap catch-up after a hitch
	{
		ControlStep();
		Accumulator -= Task.ControlDtS;
		++Steps;
	}
	if (QuitAfter > 0.0 && RaceTime >= QuitAfter)
	{
		FPlatformMisc::RequestExit(false);
	}
}

void ALORaceManager::ControlStep()
{
	TArray<float> Obs;
	const FVector2D G = Checkpoint(Opponent.NextCheckpoint);
	float A[2] = {0.f, 0.f};
	if (Opponent.Robot->BuildObservation(G.X, G.Y, Obs) && Policy.Run(Obs, A))
	{
		Advance(Opponent, Opponent.Robot->StepKinematics(A[0], A[1]));
	}
	double L = 0.0;
	double R = 0.0;
	if (bDemoPlayer)
	{
		DemoAction(L, R);
	}
	else
	{
		PlayerAction(L, R);
	}
	Advance(Player, Player.Robot->StepKinematics(L, R));
}

void ALORaceManager::Advance(FLORacer& Racer, bool bCollided)
{
	if (bCollided)
	{
		++Racer.Crashes;
		Respawn(Racer);
		return;
	}
	const FVector2D G = Checkpoint(Racer.NextCheckpoint);
	if (FVector2D::Distance(FVector2D(Racer.Robot->X(), Racer.Robot->Y()), G) <= Task.ArrivalToleranceM)
	{
		++Racer.NextCheckpoint;
		if (Racer.NextCheckpoint % Checkpoints.Num() == 0)
		{
			++Racer.Laps;
		}
	}
}

void ALORaceManager::Respawn(FLORacer& Racer)
{
	// Back to the last checkpoint reached (or the start), facing the next one.
	const bool bStart = Racer.NextCheckpoint == 0;
	const FVector Start = &Racer == &Opponent ? StartOpponent : StartPlayer;
	const FVector2D From = bStart ? FVector2D(Start.X, Start.Y) : Checkpoint(Racer.NextCheckpoint - 1);
	const FVector2D To = Checkpoint(Racer.NextCheckpoint);
	Racer.Robot->SetPose(From.X, From.Y, FMath::Atan2(To.Y - From.Y, To.X - From.X));
}

void ALORaceManager::PlayerAction(double& OutLeft, double& OutRight) const
{
	const APlayerController* PC = GetWorld()->GetFirstPlayerController();
	if (!PC)
	{
		OutLeft = OutRight = 0.0;
		return;
	}
	const double Forward = (PC->IsInputKeyDown(EKeys::W) || PC->IsInputKeyDown(EKeys::Up) ? 1.0 : 0.0) -
		(PC->IsInputKeyDown(EKeys::S) || PC->IsInputKeyDown(EKeys::Down) ? 1.0 : 0.0);
	const double Turn = (PC->IsInputKeyDown(EKeys::A) || PC->IsInputKeyDown(EKeys::Left) ? 1.0 : 0.0) -
		(PC->IsInputKeyDown(EKeys::D) || PC->IsInputKeyDown(EKeys::Right) ? 1.0 : 0.0);
	OutLeft = FMath::Clamp(Forward - 0.5 * Turn, -1.0, 1.0);
	OutRight = FMath::Clamp(Forward + 0.5 * Turn, -1.0, 1.0);
}

void ALORaceManager::DemoAction(double& OutLeft, double& OutRight) const
{
	// Scripted stand-in for a human player (README GIF): steer to the checkpoint, swerve
	// toward the clearer side when something is close ahead.
	const ALORobot* R = Player.Robot;
	const FVector2D G = Checkpoint(Player.NextCheckpoint);
	double Err = FMath::Atan2(G.Y - R->Y(), G.X - R->X()) - R->Yaw();
	Err = FMath::Atan2(FMath::Sin(Err), FMath::Cos(Err));
	double Steer = FMath::Clamp(2.0 * Err, -1.0, 1.0);
	TArray<double> Ranges;
	R->TraceRays(Ranges);
	const int32 Ahead = Ranges.Num() / 2;
	auto Clear = [&Ranges](int32 I) { return Ranges[I] < 0.0 ? 99.0 : Ranges[I]; };
	double Front = 99.0;
	for (int32 K = -2; K <= 2; ++K)
	{
		Front = FMath::Min(Front, Clear(Ahead + K));
	}
	if (Front < 0.6)
	{
		const double LeftSide = Clear(Ahead + 3) + Clear(Ahead + 4);
		const double RightSide = Clear(Ahead - 3) + Clear(Ahead - 4);
		Steer = LeftSide >= RightSide ? 1.0 : -1.0;
	}
	const double Go = FMath::Clamp(1.0 - 0.8 * FMath::Abs(Steer), 0.15, 0.8);
	OutLeft = FMath::Clamp(Go - 0.5 * Steer, -1.0, 1.0);
	OutRight = FMath::Clamp(Go + 0.5 * Steer, -1.0, 1.0);
}

TArray<FString> ALORaceManager::StatusLines() const
{
	TArray<FString> Lines;
	if (!Error.IsEmpty())
	{
		Lines.Add(FString::Printf(TEXT("Error: %s"), *Error));
		return Lines;
	}
	const int32 N = FMath::Max(1, Checkpoints.Num());
	Lines.Add(FString::Printf(TEXT("Learned Opponents  |  opponent: ONNX policy %s via %s"), *ModelName, FLOPolicy::RuntimeName));
	Lines.Add(FString::Printf(TEXT("Opponent (blue): lap %d, checkpoint %d/%d, crashes %d"),
		Opponent.Laps + 1, Opponent.NextCheckpoint % N + 1, N, Opponent.Crashes));
	Lines.Add(FString::Printf(TEXT("Player (green, %s): lap %d, checkpoint %d/%d, crashes %d"),
		bDemoPlayer ? TEXT("scripted demo controller") : TEXT("W/A/S/D"), Player.Laps + 1, Player.NextCheckpoint % N + 1, N,
		Player.Crashes));
	Lines.Add(FString::Printf(TEXT("Time %.1f s"), RaceTime));
	return Lines;
}
