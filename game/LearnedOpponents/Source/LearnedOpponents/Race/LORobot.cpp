#include "Race/LORobot.h"

#include "Components/CapsuleComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "Obs/LOObsSpec.h"
#include "Race/LOArena.h"

ALORobot::ALORobot()
{
	PrimaryActorTick.bCanEverTick = false;
	Collision = CreateDefaultSubobject<UCapsuleComponent>(TEXT("Collision"));
	RootComponent = Collision;
	Collision->SetMobility(EComponentMobility::Movable);
	Collision->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
	Collision->SetCollisionObjectType(ECC_Pawn);
	Collision->SetCollisionResponseToAllChannels(ECR_Ignore);
	Collision->SetCollisionResponseToChannel(ECC_WorldStatic, ECR_Block);

	Body = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Body"));
	Body->SetupAttachment(Collision);
	Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	Nose = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Nose"));
	Nose->SetupAttachment(Collision);
	Nose->SetCollisionEnabled(ECollisionEnabled::NoCollision);
}

void ALORobot::Configure(const FLOTaskData& Task, const FLinearColor& Color)
{
	TaskData = Task;
	RayAngles = LOObs::RayAnglesRad(Task.Spec);
	const double RCm = Task.RobotRadiusM * 100.0;
	Collision->SetCapsuleSize(RCm, RCm, false);       // a sphere: its widest circle sits at ray height

	Body->SetStaticMesh(LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder")));
	Body->SetRelativeScale3D(FVector(2 * RCm / 100.0, 2 * RCm / 100.0, 0.12));
	Body->SetMaterial(0, ALOArena::ColorMaterial(this, Color));
	Nose->SetStaticMesh(LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube")));
	Nose->SetRelativeLocation(FVector(RCm * 0.7, 0, 7.0));
	Nose->SetRelativeScale3D(FVector(0.08, 0.06, 0.04));
	Nose->SetMaterial(0, ALOArena::ColorMaterial(this, FLinearColor::White));
}

void ALORobot::SyncActor(bool bSweep, FHitResult* OutHit)
{
	const FVector Loc = LOObs::SpecToUnrealLocation(PoseX, PoseY, TaskData.RayHeightM * 100.0);
	const FRotator Rot(0.0, LOObs::SpecToUnrealYawDeg(PoseYaw), 0.0);
	SetActorLocationAndRotation(Loc, Rot, bSweep, OutHit, bSweep ? ETeleportType::None : ETeleportType::TeleportPhysics);
}

void ALORobot::SetPose(double X, double Y, double Yaw)
{
	PoseX = X;
	PoseY = Y;
	PoseYaw = Yaw;
	VelLinear = 0.0;
	VelAngular = 0.0;
	SyncActor(false, nullptr);
}

bool ALORobot::StepKinematics(double WheelLeft, double WheelRight)
{
	const FLOSpecConfig& S = TaskData.Spec;
	const double Wl = FMath::Clamp(WheelLeft, -1.0, 1.0) * S.MaxWheelSpeed;
	const double Wr = FMath::Clamp(WheelRight, -1.0, 1.0) * S.MaxWheelSpeed;
	VelLinear = S.WheelRadiusM * (Wl + Wr) / 2.0;
	VelAngular = S.WheelRadiusM * (Wr - Wl) / S.TrackWidthM;
	const int32 N = TaskData.Substeps();
	const double Dt = TaskData.ControlDtS / N;
	for (int32 K = 0; K < N; ++K)
	{
		// Exact unicycle integration over the substep.
		const double NewYaw = PoseYaw + VelAngular * Dt;
		if (FMath::Abs(VelAngular) > 1e-9)
		{
			PoseX += VelLinear / VelAngular * (FMath::Sin(NewYaw) - FMath::Sin(PoseYaw));
			PoseY -= VelLinear / VelAngular * (FMath::Cos(NewYaw) - FMath::Cos(PoseYaw));
		}
		else
		{
			PoseX += VelLinear * FMath::Cos(PoseYaw) * Dt;
			PoseY += VelLinear * FMath::Sin(PoseYaw) * Dt;
		}
		PoseYaw = FMath::UnwindRadians(NewYaw);
		FHitResult Hit;
		SyncActor(true, &Hit);
		if (Hit.bBlockingHit || Hit.bStartPenetrating)
		{
			const FVector2D At = LOObs::UnrealToSpecXY(GetActorLocation().X, GetActorLocation().Y);
			PoseX = At.X;
			PoseY = At.Y;
			return true;
		}
	}
	return false;
}

void ALORobot::TraceRays(TArray<double>& OutRangesM) const
{
	UWorld* World = GetWorld();
	const FVector Start = GetActorLocation();
	const double YawDeg = LOObs::SpecToUnrealYawDeg(PoseYaw);
	// Slightly longer than max range so a hit exactly at max range is still a hit.
	const double LengthCm = (TaskData.Spec.MaxRangeM + 0.01) * 100.0;
	FCollisionQueryParams Params(SCENE_QUERY_STAT(LORays), false, this);
	const FCollisionObjectQueryParams Objects(ECC_WorldStatic);
	OutRangesM.SetNumUninitialized(RayAngles.Num());
	for (int32 I = 0; I < RayAngles.Num(); ++I)
	{
		const FVector Dir = FRotator(0.0, YawDeg + LOObs::SpecRayAngleToUnrealYawOffsetDeg(RayAngles[I]), 0.0).Vector();
		FHitResult Hit;
		const bool bHit = World->LineTraceSingleByObjectType(Hit, Start, Start + Dir * LengthCm, Objects, Params);
		OutRangesM[I] = bHit ? Hit.Distance / 100.0 : -1.0;
	}
}

bool ALORobot::BuildObservation(double GoalX, double GoalY, TArray<float>& OutObs) const
{
	TArray<double> Ranges;
	TraceRays(Ranges);
	return LOObs::BuildObservation(TaskData.Spec, Ranges, PoseX, PoseY, PoseYaw, GoalX, GoalY, VelLinear, VelAngular, OutObs);
}
