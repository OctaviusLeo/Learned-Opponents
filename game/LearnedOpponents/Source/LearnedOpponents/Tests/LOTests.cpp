// Automation tests, all under the "LearnedOpponents." prefix (scripts/run_ue_tests.ps1).

#include "Dom/JsonObject.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "HAL/FileManager.h"
#include "Inference/LONpy.h"
#include "Inference/LOPolicy.h"
#include "Misc/AutomationTest.h"
#include "Misc/Paths.h"
#include "Obs/LOGameData.h"
#include "Obs/LOObsSpec.h"
#include "Race/LOArena.h"
#include "Race/LORobot.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace
{
	constexpr EAutomationTestFlags TestFlags = EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter;

	FLOTaskData TaskOrDefault()
	{
		FLOTaskData Task;
		FString Err;
		LOData::LoadTask(LOData::ResultsPath(TEXT("game/task.json")), Task, Err);
		return Task;
	}

	// A temporary game world with physics, for raycast and sweep tests.
	struct FLOTestWorld
	{
		UWorld* World = nullptr;

		FLOTestWorld()
		{
			World = UWorld::CreateWorld(EWorldType::Game, false, TEXT("LOTestWorld"));
			FWorldContext& Ctx = GEngine->CreateNewWorldContext(EWorldType::Game);
			Ctx.SetCurrentWorld(World);
			World->InitializeActorsForPlay(FURL());
			World->BeginPlay();
		}

		void Tick(int32 Frames)
		{
			for (int32 I = 0; I < Frames; ++I)
			{
				World->Tick(LEVELTICK_All, 1.f / 60.f);
			}
		}

		~FLOTestWorld()
		{
			GEngine->DestroyWorldContext(World);
			World->DestroyWorld(false);
		}
	};

	// Robot at the origin; a 0.6 m box 2 m ahead, a 0.4 m cylinder 1.5 m left, a 0.3 m box 1 m right
	// (the same scene as learned_opponents.env.nav_env.known_distance_layout).
	FLOLayout KnownDistanceLayout(double Yaw)
	{
		FLOLayout L;
		L.Obstacles.Add({ELOObstacleKind::Box, 2.0, 0.0, 0.6});
		L.Obstacles.Add({ELOObstacleKind::Cylinder, 0.0, 1.5, 0.4});
		L.Obstacles.Add({ELOObstacleKind::Box, 0.0, -1.0, 0.3});
		L.StartYaw = Yaw;
		L.GoalX = 3.0;
		L.GoalY = 3.0;
		return L;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLOSmokeSpecDefaults, "LearnedOpponents.Smoke.SpecDefaults", TestFlags)
bool FLOSmokeSpecDefaults::RunTest(const FString& Parameters)
{
	const FLOSpecConfig Cfg;
	TestEqual(TEXT("observation dimension"), Cfg.ObsDim(), 29);
	TestEqual(TEXT("v_max"), Cfg.VMax(), 1.0, 1e-12);
	TestEqual(TEXT("omega_max"), Cfg.OmegaMax(), 2.0 * 0.05 * 20.0 / 0.30, 1e-12);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLOObsKnownRaycasts, "LearnedOpponents.Obs.KnownRaycasts", TestFlags)
bool FLOObsKnownRaycasts::RunTest(const FString& Parameters)
{
	const FLOSpecConfig Cfg;
	const TArray<double> Angles = LOObs::RayAnglesRad(Cfg);
	TestEqual(TEXT("ray 0 points back"), FMath::RadiansToDegrees(Angles[0]), -180.0, 1e-9);
	TestEqual(TEXT("ray 6 points right"), FMath::RadiansToDegrees(Angles[6]), -90.0, 1e-9);
	TestEqual(TEXT("ray 12 points ahead"), FMath::RadiansToDegrees(Angles[12]), 0.0, 1e-9);
	TestEqual(TEXT("ray 18 points left"), FMath::RadiansToDegrees(Angles[18]), 90.0, 1e-9);

	// Miss, beyond range, infinity -> 1; a 2 m hit -> 0.5.
	TArray<double> Ranges;
	Ranges.Init(2.0, Cfg.NRays);
	Ranges[0] = -1.0;
	Ranges[1] = 7.5;
	Ranges[2] = TNumericLimits<double>::Max() * 2.0;
	TArray<float> Obs;
	TestTrue(TEXT("build"), LOObs::BuildObservation(Cfg, Ranges, 0, 0, 0, 1, 0, 0, 0, Obs));
	TestEqual(TEXT("obs length"), Obs.Num(), 29);
	TestEqual(TEXT("miss -> 1"), Obs[0], 1.f);
	TestEqual(TEXT("beyond range -> 1"), Obs[1], 1.f);
	TestEqual(TEXT("infinite -> 1"), Obs[2], 1.f);
	TestEqual(TEXT("2 m -> 0.5"), Obs[3], 0.5f);

	// Robot at (1, 1) facing +y; goal 2 m ahead -> direction (1, 0) in the robot frame.
	Ranges.Init(4.0, Cfg.NRays);
	LOObs::BuildObservation(Cfg, Ranges, 1.0, 1.0, UE_DOUBLE_PI / 2.0, 1.0, 3.0, 0.5, -1.0, Obs);
	TestEqual(TEXT("goal_dir_x"), Obs[24], 1.f, 1e-6f);
	TestEqual(TEXT("goal_dir_y"), Obs[25], 0.f, 1e-6f);
	TestEqual(TEXT("goal_distance"), Obs[26], float(2.0 / Cfg.ArenaDiagonalM()), 1e-6f);
	TestEqual(TEXT("vel_linear"), Obs[27], 0.5f, 1e-6f);
	TestEqual(TEXT("vel_angular"), Obs[28], float(-1.0 / Cfg.OmegaMax()), 1e-6f);
	// Goal behind-left in the world turns into robot-frame +y.
	LOObs::BuildObservation(Cfg, Ranges, 1.0, 1.0, UE_DOUBLE_PI / 2.0, 0.0, 1.0, 0, 0, Obs);
	TestEqual(TEXT("goal left: x"), Obs[24], 0.f, 1e-6f);
	TestEqual(TEXT("goal left: y"), Obs[25], 1.f, 1e-6f);
	// Velocities clip to [-1, 1].
	LOObs::BuildObservation(Cfg, Ranges, 0, 0, 0, 1, 0, 9.0, -99.0, Obs);
	TestEqual(TEXT("clip vel_linear"), Obs[27], 1.f);
	TestEqual(TEXT("clip vel_angular"), Obs[28], -1.f);
	// Unreal conversion: x forward, y right, yaw clockwise.
	TestEqual(TEXT("unreal y mirrored"), LOObs::UnrealToSpecXY(250.0, -100.0).Y, 1.0, 1e-12);
	TestEqual(TEXT("unreal yaw"), LOObs::UnrealToSpecYaw(90.0), -UE_DOUBLE_PI / 2.0, 1e-12);
	TestEqual(TEXT("ray offset"), LOObs::SpecRayAngleToUnrealYawOffsetDeg(UE_DOUBLE_PI / 2.0), -90.0, 1e-9);
	TestFalse(TEXT("wrong ray count rejected"), LOObs::BuildObservation(Cfg, TArray<double>{1.0}, 0, 0, 0, 1, 0, 0, 0, Obs));
	return true;
}

// Feature: learned-opponents, Property 3: Observation construction is engine-agnostic.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLOObsEngineAgnostic, "LearnedOpponents.Obs.EngineAgnosticFixtures", TestFlags)
bool FLOObsEngineAgnostic::RunTest(const FString& Parameters)
{
	FString Err;
	const TSharedPtr<FJsonObject> Root = LOData::LoadJson(LOData::ResultsPath(TEXT("fixtures/obs_cases.json")), Err);
	if (!TestTrue(FString::Printf(TEXT("fixtures load: %s"), *Err), Root.IsValid()))
	{
		return false;
	}
	FLOSpecConfig Cfg;
	const TSharedPtr<FJsonObject>* Spec = nullptr;
	TestTrue(TEXT("spec"), Root->TryGetObjectField(TEXT("spec"), Spec) && LOData::ParseSpec(*Spec, Cfg));
	const double Tol = Root->GetNumberField(TEXT("tolerance"));
	const TArray<TSharedPtr<FJsonValue>>& Cases = Root->GetArrayField(TEXT("cases"));
	TestTrue(TEXT(">= 100 cases"), Cases.Num() >= 100);
	double Worst = 0.0;
	for (int32 C = 0; C < Cases.Num(); ++C)
	{
		const TSharedPtr<FJsonObject> Case = Cases[C]->AsObject();
		TArray<double> Ranges;
		TArray<double> RangesCm;
		for (const TSharedPtr<FJsonValue>& V : Case->GetArrayField(TEXT("ranges_m")))
		{
			Ranges.Add(V->AsNumber());
		}
		const TSharedPtr<FJsonObject> U = Case->GetObjectField(TEXT("unreal"));
		for (const TSharedPtr<FJsonValue>& V : U->GetArrayField(TEXT("ranges_cm")))
		{
			RangesCm.Add(V->AsNumber());
		}
		TArray<float> Expected;
		for (const TSharedPtr<FJsonValue>& V : Case->GetArrayField(TEXT("expected")))
		{
			Expected.Add(static_cast<float>(V->AsNumber()));
		}
		TArray<float> FromSpec;
		TArray<float> FromUnreal;
		LOObs::BuildObservation(Cfg, Ranges, Case->GetNumberField(TEXT("robot_x")), Case->GetNumberField(TEXT("robot_y")),
			Case->GetNumberField(TEXT("robot_yaw")), Case->GetNumberField(TEXT("goal_x")), Case->GetNumberField(TEXT("goal_y")),
			Case->GetNumberField(TEXT("vel_linear")), Case->GetNumberField(TEXT("vel_angular")), FromSpec);
		LOObs::BuildObservationFromUnreal(Cfg, RangesCm,
			FVector(U->GetNumberField(TEXT("robot_x_cm")), U->GetNumberField(TEXT("robot_y_cm")), 0.0),
			U->GetNumberField(TEXT("robot_yaw_deg")),
			FVector(U->GetNumberField(TEXT("goal_x_cm")), U->GetNumberField(TEXT("goal_y_cm")), 0.0),
			U->GetNumberField(TEXT("speed_cm_s")), U->GetNumberField(TEXT("yaw_rate_deg_s")), FromUnreal);
		if (!TestEqual(TEXT("vector length"), FromSpec.Num(), Expected.Num()) || FromUnreal.Num() != Expected.Num())
		{
			return false;
		}
		for (int32 I = 0; I < Expected.Num(); ++I)
		{
			Worst = FMath::Max(Worst, FMath::Max(FMath::Abs(double(FromSpec[I]) - Expected[I]),
				FMath::Abs(double(FromUnreal[I]) - Expected[I])));
		}
	}
	AddInfo(FString::Printf(TEXT("%d cases, worst |C++ - Python| = %.3g (tolerance %.0e)"), Cases.Num(), Worst, Tol));
	TestTrue(TEXT("every case within tolerance"), Worst <= Tol);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLOObsTracedDistances, "LearnedOpponents.Obs.TracedKnownDistances", TestFlags)
bool FLOObsTracedDistances::RunTest(const FString& Parameters)
{
	const FLOTaskData Task = TaskOrDefault();
	FLOTestWorld W;
	ALOArena* Arena = W.World->SpawnActor<ALOArena>();
	Arena->Build(Task);
	ALORobot* Robot = W.World->SpawnActor<ALORobot>();
	Robot->Configure(Task, FLinearColor::Blue);
	struct FCaseDef { double Yaw; TArray<TPair<int32, double>> Expect; };
	const TArray<FCaseDef> Defs = {
		{0.0, {{12, 1.7}, {18, 1.3}, {6, 0.85}, {0, 4.0}}},
		{UE_DOUBLE_PI / 2.0, {{12, 1.3}, {6, 1.7}, {0, 0.85}}},
	};
	for (const FCaseDef& D : Defs)
	{
		const FLOLayout L = KnownDistanceLayout(D.Yaw);
		Arena->ApplyLayout(L);
		Robot->SetPose(0.0, 0.0, D.Yaw);
		W.Tick(3);
		TArray<double> Ranges;
		Robot->TraceRays(Ranges);
		TestEqual(TEXT("ray count"), Ranges.Num(), Task.Spec.NRays);
		for (const TPair<int32, double>& E : D.Expect)
		{
			TestEqual(FString::Printf(TEXT("yaw %.2f ray %d"), D.Yaw, E.Key), Ranges[E.Key], E.Value, 1e-3);
		}
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLORaceKinematics, "LearnedOpponents.Race.KinematicsAndCollision", TestFlags)
bool FLORaceKinematics::RunTest(const FString& Parameters)
{
	const FLOTaskData Task = TaskOrDefault();
	FLOTestWorld W;
	ALOArena* Arena = W.World->SpawnActor<ALOArena>();
	Arena->Build(Task);
	FLOLayout L;
	L.Obstacles.Add({ELOObstacleKind::Box, 1.0, 0.0, 0.6});
	Arena->ApplyLayout(L);
	ALORobot* Robot = W.World->SpawnActor<ALORobot>();
	Robot->Configure(Task, FLinearColor::Blue);
	W.Tick(3);
	// Half speed straight ahead for 1 s in open space: 0.5 m.
	Robot->SetPose(-3.0, -2.0, 0.0);
	for (int32 I = 0; I < 20; ++I)
	{
		TestFalse(TEXT("no collision in open space"), Robot->StepKinematics(0.5, 0.5));
	}
	TestEqual(TEXT("x after 1 s"), Robot->X(), -2.5, 1e-6);
	// Spin counter-clockwise at 0.3 of omega_max for 1 s.
	Robot->SetPose(-3.0, -2.0, 0.0);
	for (int32 I = 0; I < 20; ++I)
	{
		Robot->StepKinematics(-0.3, 0.3);
	}
	TestEqual(TEXT("yaw after 1 s"), Robot->Yaw(), 0.3 * Task.Spec.OmegaMax(), 1e-6);
	// Drive at the box from 2 m away: collision once the footprint reaches its face (x = 0.55).
	Robot->SetPose(-1.0, 0.0, 0.0);
	int32 Steps = 0;
	bool bHit = false;
	while (!bHit && Steps < 200)
	{
		bHit = Robot->StepKinematics(0.6, 0.6);
		++Steps;
	}
	TestTrue(TEXT("collision detected"), bHit);
	TestEqual(TEXT("stopped at the box face"), Robot->X(), 1.0 - 0.3 - Task.RobotRadiusM, 0.02);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLOInferenceParity, "LearnedOpponents.Inference.ParityAllModels", TestFlags)
bool FLOInferenceParity::RunTest(const FString& Parameters)
{
	TArray<FString> Files;
	IFileManager::Get().FindFiles(Files, *LOData::ResultsPath(TEXT("models/*_seed*.onnx")), true, false);
	Files.Sort();
	if (!TestTrue(TEXT("exported models found under results/models"), Files.Num() > 0))
	{
		return false;
	}
	for (const FString& File : Files)
	{
		const FString Name = FPaths::GetBaseFilename(File);
		TArray<float> Obs;
		TArray<float> Act;
		TArray<int64> ObsShape;
		TArray<int64> ActShape;
		FString Err;
		const bool bLoaded = LONpy::LoadFloat32(LOData::ResultsPath(FString::Printf(TEXT("parity/%s/observations.npy"), *Name)), Obs, ObsShape, Err) &&
			LONpy::LoadFloat32(LOData::ResultsPath(FString::Printf(TEXT("parity/%s/actions.npy"), *Name)), Act, ActShape, Err);
		if (!TestTrue(FString::Printf(TEXT("%s fixtures: %s"), *Name, *Err), bLoaded && ObsShape.Num() == 2 && ActShape.Num() == 2))
		{
			continue;
		}
		FLOPolicy Policy;
		const int32 Dim = static_cast<int32>(ObsShape[1]);
		if (!TestTrue(FString::Printf(TEXT("%s loads in %s: %s"), *Name, FLOPolicy::RuntimeName, *Err),
			Policy.Load(LOData::ResultsPath(FString::Printf(TEXT("models/%s.onnx"), *Name)), Dim, Err)))
		{
			continue;
		}
		double Worst = 0.0;
		int64 WorstIndex = -1;
		for (int64 I = 0; I < ObsShape[0]; ++I)
		{
			float A[2];
			if (!Policy.Run(TConstArrayView<float>(Obs.GetData() + I * Dim, Dim), A))
			{
				AddError(FString::Printf(TEXT("%s: RunSync failed at observation %lld"), *Name, I));
				break;
			}
			for (int32 K = 0; K < 2; ++K)
			{
				const double D = FMath::Abs(double(A[K]) - Act[I * 2 + K]);
				if (D > Worst)
				{
					Worst = D;
					WorstIndex = I;
				}
			}
		}
		AddInfo(FString::Printf(TEXT("%s: %lld observations, max |NNE - Python| %.3g at observation %lld"), *Name, ObsShape[0],
			Worst, WorstIndex));
		TestTrue(FString::Printf(TEXT("%s within 1e-4"), *Name), Worst <= 1e-4);
	}
	return true;
}

#endif
