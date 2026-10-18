#include "Transfer/LOTransferHarness.h"

#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformTime.h"
#include "LearnedOpponents.h"
#include "Misc/EngineVersion.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Race/LOArena.h"
#include "Race/LORobot.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

ALOTransferHarness::ALOTransferHarness()
{
	PrimaryActorTick.bCanEverTick = true;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

void ALOTransferHarness::BeginPlay()
{
	Super::BeginPlay();
	FString Err;
	if (!LOData::LoadTask(LOData::ResultsPath(TEXT("game/task.json")), Task, Err))
	{
		UE_LOG(LogLearnedOpponents, Error, TEXT("transfer: %s"), *Err);
		Finish(false);
		return;
	}
	EpisodesPath = LOData::CmdValue(TEXT("LOEpisodes"), LOData::ResultsPath(TEXT("transfer/episodes.json")));
	if (!LOData::LoadLayouts(EpisodesPath, Layouts, Err) || Layouts.Num() == 0)
	{
		UE_LOG(LogLearnedOpponents, Error, TEXT("transfer: %s"), *Err);
		Finish(false);
		return;
	}
	const FString Models = LOData::CmdValue(TEXT("LOModels"), TEXT(""));
	if (Models.IsEmpty())
	{
		TArray<FString> Files;
		IFileManager::Get().FindFiles(Files, *LOData::ResultsPath(TEXT("models/*_seed*.onnx")), true, false);
		Files.Sort();
		for (const FString& F : Files)
		{
			ModelNames.Add(FPaths::GetBaseFilename(F));
		}
	}
	else
	{
		Models.ParseIntoArray(ModelNames, TEXT(","), true);
	}
	OutPath = LOData::CmdValue(TEXT("LOOut"), LOData::ResultsPath(TEXT("transfer/game_results.json")));
	Arena = GetWorld()->SpawnActor<ALOArena>();
	Arena->Build(Task);
	Robot = GetWorld()->SpawnActor<ALORobot>();
	Robot->Configure(Task, FLinearColor(0.10f, 0.35f, 0.90f));
	StartSeconds = FPlatformTime::Seconds();
	UE_LOG(LogLearnedOpponents, Display, TEXT("transfer: %d models x %d layouts from %s"), ModelNames.Num(), Layouts.Num(),
		*EpisodesPath);
}

FString ALOTransferHarness::RunEpisode(ALORobot* InRobot, FLOPolicy& InPolicy, const FLOLayout& Layout, int32& OutSteps)
{
	const FLOTaskData& T = InRobot->Task();
	InRobot->SetPose(Layout.StartX, Layout.StartY, Layout.StartYaw);
	TArray<float> Obs;
	for (int32 Step = 1; Step <= T.EpisodeLimit; ++Step)
	{
		float A[2] = {0.f, 0.f};
		if (!InRobot->BuildObservation(Layout.GoalX, Layout.GoalY, Obs) || !InPolicy.Run(Obs, A))
		{
			OutSteps = Step;
			return TEXT("error");
		}
		const bool bCollided = InRobot->StepKinematics(A[0], A[1]);
		const double Dist = FVector2D::Distance(FVector2D(InRobot->X(), InRobot->Y()), FVector2D(Layout.GoalX, Layout.GoalY));
		OutSteps = Step;
		if (bCollided)
		{
			return TEXT("collision");
		}
		if (Dist <= T.ArrivalToleranceM)
		{
			return TEXT("success");
		}
	}
	return TEXT("timeout");
}

void ALOTransferHarness::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (bDone || !Arena)
	{
		return;
	}
	if (ModelIndex >= ModelNames.Num())
	{
		Finish(true);
		return;
	}
	if (EpisodeIndex == 0 && !Policy.IsLoaded())
	{
		FString Err;
		const FString Path = LOData::ResultsPath(FString::Printf(TEXT("models/%s.onnx"), *ModelNames[ModelIndex]));
		if (!Policy.Load(Path, Task.Spec.ObsDim(), Err))
		{
			UE_LOG(LogLearnedOpponents, Error, TEXT("transfer: %s"), *Err);
			Finish(false);
			return;
		}
	}
	if (!bLayoutApplied)
	{
		Arena->ApplyLayout(Layouts[EpisodeIndex]);
		bLayoutApplied = true;
		WaitFrames = 2;
		return;
	}
	if (WaitFrames-- > 0)
	{
		return;
	}
	int32 EpSteps = 0;
	const FString Reason = RunEpisode(Robot, Policy, Layouts[EpisodeIndex], EpSteps);
	if (Reason == TEXT("error"))
	{
		UE_LOG(LogLearnedOpponents, Error, TEXT("transfer: inference failed on %s episode %d"), *ModelNames[ModelIndex], EpisodeIndex);
		Finish(false);
		return;
	}
	Reasons.Add(Reason);
	Steps.Add(EpSteps);
	bLayoutApplied = false;
	if (++EpisodeIndex >= Layouts.Num())
	{
		UE_LOG(LogLearnedOpponents, Display, TEXT("transfer: %s done"), *ModelNames[ModelIndex]);
		EpisodeIndex = 0;
		++ModelIndex;
		Policy = FLOPolicy();
	}
}

void ALOTransferHarness::Finish(bool bOk)
{
	bDone = true;
	if (bOk)
	{
		TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
		Root->SetStringField(TEXT("runtime"), FLOPolicy::RuntimeName);
		Root->SetStringField(TEXT("engine_version"), FEngineVersion::Current().ToString());
		Root->SetStringField(TEXT("episodes_file"), EpisodesPath);
		Root->SetStringField(TEXT("task_commit"), Task.Commit);
		Root->SetNumberField(TEXT("control_dt_s"), Task.ControlDtS);
		Root->SetNumberField(TEXT("wall_s"), FPlatformTime::Seconds() - StartSeconds);
		TSharedRef<FJsonObject> PerModel = MakeShared<FJsonObject>();
		for (int32 M = 0; M < ModelNames.Num(); ++M)
		{
			TArray<TSharedPtr<FJsonValue>> Eps;
			for (int32 E = 0; E < Layouts.Num(); ++E)
			{
				const int32 K = M * Layouts.Num() + E;
				TSharedRef<FJsonObject> Ep = MakeShared<FJsonObject>();
				Ep->SetNumberField(TEXT("index"), E);
				Ep->SetStringField(TEXT("reason"), Reasons[K]);
				Ep->SetNumberField(TEXT("steps"), Steps[K]);
				Eps.Add(MakeShared<FJsonValueObject>(Ep));
			}
			PerModel->SetArrayField(ModelNames[M], Eps);
		}
		Root->SetObjectField(TEXT("models"), PerModel);
		FString Text;
		const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Text);
		FJsonSerializer::Serialize(Root, Writer);
		FFileHelper::SaveStringToFile(Text, *OutPath);
		UE_LOG(LogLearnedOpponents, Display, TEXT("transfer: wrote %s (%d episodes)"), *OutPath, Reasons.Num());
	}
	FPlatformMisc::RequestExitWithStatus(false, bOk ? 0 : 1);
}
