#include "Obs/LOGameData.h"

#include "Dom/JsonObject.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace LOData
{
	FString RepoRoot()
	{
		FString Root = FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectDir(), TEXT("../..")));
		FPaths::NormalizeDirectoryName(Root);
		return Root;
	}

	FString ResultsPath(const FString& Relative)
	{
		return FPaths::Combine(RepoRoot(), TEXT("results"), Relative);
	}

	TSharedPtr<FJsonObject> LoadJson(const FString& Path, FString& OutError)
	{
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			OutError = FString::Printf(TEXT("cannot read %s"), *Path);
			return nullptr;
		}
		TSharedPtr<FJsonObject> Obj;
		const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Obj) || !Obj.IsValid())
		{
			OutError = FString::Printf(TEXT("invalid JSON in %s"), *Path);
			return nullptr;
		}
		return Obj;
	}

	bool ParseSpec(const TSharedPtr<FJsonObject>& Obj, FLOSpecConfig& Out)
	{
		if (!Obj.IsValid())
		{
			return false;
		}
		double NRays = 0.0;
		bool bOk = Obj->TryGetNumberField(TEXT("n_rays"), NRays);
		bOk &= Obj->TryGetNumberField(TEXT("ray_fov_deg"), Out.RayFovDeg);
		bOk &= Obj->TryGetNumberField(TEXT("max_range_m"), Out.MaxRangeM);
		bOk &= Obj->TryGetNumberField(TEXT("arena_size_m"), Out.ArenaSizeM);
		bOk &= Obj->TryGetNumberField(TEXT("wheel_radius_m"), Out.WheelRadiusM);
		bOk &= Obj->TryGetNumberField(TEXT("track_width_m"), Out.TrackWidthM);
		bOk &= Obj->TryGetNumberField(TEXT("max_wheel_speed"), Out.MaxWheelSpeed);
		Out.NRays = static_cast<int32>(NRays);
		return bOk && Out.NRays >= 2;
	}

	bool LoadTask(const FString& Path, FLOTaskData& Out, FString& OutError)
	{
		const TSharedPtr<FJsonObject> Obj = LoadJson(Path, OutError);
		if (!Obj.IsValid())
		{
			return false;
		}
		const TSharedPtr<FJsonObject>* Spec = nullptr;
		if (!Obj->TryGetObjectField(TEXT("spec"), Spec) || !ParseSpec(*Spec, Out.Spec))
		{
			OutError = FString::Printf(TEXT("%s: missing or invalid 'spec'"), *Path);
			return false;
		}
		double Limit = 0.0;
		double MaxObs = 0.0;
		bool bOk = Obj->TryGetNumberField(TEXT("robot_radius_m"), Out.RobotRadiusM);
		bOk &= Obj->TryGetNumberField(TEXT("ray_height_m"), Out.RayHeightM);
		bOk &= Obj->TryGetNumberField(TEXT("control_dt_s"), Out.ControlDtS);
		bOk &= Obj->TryGetNumberField(TEXT("physics_dt_s"), Out.PhysicsDtS);
		bOk &= Obj->TryGetNumberField(TEXT("arrival_tolerance_m"), Out.ArrivalToleranceM);
		bOk &= Obj->TryGetNumberField(TEXT("wall_height_m"), Out.WallHeightM);
		bOk &= Obj->TryGetNumberField(TEXT("obstacle_height_m"), Out.ObstacleHeightM);
		bOk &= Obj->TryGetNumberField(TEXT("episode_limit"), Limit);
		bOk &= Obj->TryGetNumberField(TEXT("max_obstacles"), MaxObs);
		Obj->TryGetStringField(TEXT("commit"), Out.Commit);
		Out.EpisodeLimit = static_cast<int32>(Limit);
		Out.MaxObstacles = static_cast<int32>(MaxObs);
		if (!bOk)
		{
			OutError = FString::Printf(TEXT("%s: missing task fields"), *Path);
		}
		return bOk;
	}

	static bool ReadXY(const TSharedPtr<FJsonObject>& Obj, const TCHAR* Field, double& X, double& Y)
	{
		const TSharedPtr<FJsonObject>* P = nullptr;
		return Obj->TryGetObjectField(Field, P) && (*P)->TryGetNumberField(TEXT("x"), X) &&
			(*P)->TryGetNumberField(TEXT("y"), Y);
	}

	bool ParseLayout(const TSharedPtr<FJsonObject>& Episode, FLOLayout& Out)
	{
		if (!Episode.IsValid())
		{
			return false;
		}
		const TSharedPtr<FJsonObject>* Start = nullptr;
		if (!Episode->TryGetObjectField(TEXT("start"), Start) || !(*Start)->TryGetNumberField(TEXT("yaw"), Out.StartYaw))
		{
			return false;
		}
		if (!ReadXY(Episode, TEXT("start"), Out.StartX, Out.StartY) || !ReadXY(Episode, TEXT("goal"), Out.GoalX, Out.GoalY))
		{
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* Obstacles = nullptr;
		if (!Episode->TryGetArrayField(TEXT("obstacles"), Obstacles))
		{
			return false;
		}
		Out.Obstacles.Reset();
		for (const TSharedPtr<FJsonValue>& V : *Obstacles)
		{
			const TSharedPtr<FJsonObject> O = V->AsObject();
			FLOObstacle Ob;
			FString Kind;
			if (!O.IsValid() || !O->TryGetStringField(TEXT("kind"), Kind) || !O->TryGetNumberField(TEXT("x"), Ob.X) ||
				!O->TryGetNumberField(TEXT("y"), Ob.Y) || !O->TryGetNumberField(TEXT("size"), Ob.Size))
			{
				return false;
			}
			Ob.Kind = Kind == TEXT("box") ? ELOObstacleKind::Box : Kind == TEXT("cylinder") ? ELOObstacleKind::Cylinder
				: ELOObstacleKind::Inactive;
			if (Ob.Kind == ELOObstacleKind::Inactive)
			{
				return false;
			}
			Out.Obstacles.Add(Ob);
		}
		return true;
	}

	bool LoadLayouts(const FString& Path, TArray<FLOLayout>& Out, FString& OutError)
	{
		const TSharedPtr<FJsonObject> Obj = LoadJson(Path, OutError);
		const TArray<TSharedPtr<FJsonValue>>* Episodes = nullptr;
		if (!Obj.IsValid() || !Obj->TryGetArrayField(TEXT("episodes"), Episodes))
		{
			OutError = OutError.IsEmpty() ? FString::Printf(TEXT("%s: missing 'episodes'"), *Path) : OutError;
			return false;
		}
		Out.Reset();
		for (int32 I = 0; I < Episodes->Num(); ++I)
		{
			FLOLayout L;
			if (!ParseLayout((*Episodes)[I]->AsObject(), L))
			{
				OutError = FString::Printf(TEXT("%s: episode %d is malformed"), *Path, I);
				return false;
			}
			Out.Add(MoveTemp(L));
		}
		return true;
	}

	FString CmdValue(const TCHAR* Key, const FString& Default)
	{
		FString Value;
		return FParse::Value(FCommandLine::Get(), *FString::Printf(TEXT("-%s="), Key), Value, false) ? Value : Default;
	}
}
