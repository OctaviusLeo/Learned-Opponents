// A kinematic differential-drive robot with the same wheel radius, track width, and wheel speed
// limit as config/task.yaml. Its collision is a sphere of robot_radius_m centred at ray height,
// moved with sweeps, so first contact with a wall or obstacle footprint ends the episode as the
// MuJoCo footprint check does. Pose and velocities are kept in the spec frame.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Obs/LOGameData.h"
#include "LORobot.generated.h"

class UCapsuleComponent;
class UStaticMeshComponent;

UCLASS()
class LEARNEDOPPONENTS_API ALORobot : public AActor
{
	GENERATED_BODY()

public:
	ALORobot();

	void Configure(const FLOTaskData& Task, const FLinearColor& Color);

	// Teleport (no sweep) to a spec-frame pose and stop.
	void SetPose(double X, double Y, double Yaw);

	// Apply normalized wheel commands for one control step (Task.Substeps() swept substeps).
	// Returns true if the robot touched a wall or obstacle.
	bool StepKinematics(double WheelLeft, double WheelRight);

	// Rangefinder readings in metres in ray index order; -1 where a ray hits nothing.
	void TraceRays(TArray<double>& OutRangesM) const;

	// The Obs_Action_Spec observation for a goal at (GoalX, GoalY), from in-engine raycasts.
	bool BuildObservation(double GoalX, double GoalY, TArray<float>& OutObs) const;

	double X() const { return PoseX; }
	double Y() const { return PoseY; }
	double Yaw() const { return PoseYaw; }
	double LinearVelocity() const { return VelLinear; }
	double AngularVelocity() const { return VelAngular; }
	const FLOTaskData& Task() const { return TaskData; }

private:
	void SyncActor(bool bSweep, FHitResult* OutHit);

	UPROPERTY()
	TObjectPtr<UCapsuleComponent> Collision;
	UPROPERTY()
	TObjectPtr<UStaticMeshComponent> Body;
	UPROPERTY()
	TObjectPtr<UStaticMeshComponent> Nose;

	FLOTaskData TaskData;
	TArray<double> RayAngles;
	double PoseX = 0.0;
	double PoseY = 0.0;
	double PoseYaw = 0.0;
	double VelLinear = 0.0;
	double VelAngular = 0.0;
};
