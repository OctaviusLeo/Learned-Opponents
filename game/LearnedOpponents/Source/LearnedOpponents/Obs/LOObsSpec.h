// In-engine Obs_Action_Spec: the C++ mirror of learned_opponents/spec/obs_action_spec.py.
//
// Everything here works in the spec frame (right-handed, x forward, y left, metres, radians,
// counter-clockwise positive). The Unreal* helpers convert Unreal's left-handed centimetre
// frame (x forward, y right, yaw clockwise in degrees) first, so the only intended difference
// from training is the raycast source. Arithmetic is in double and the vector is rounded to
// float at the end, as the Python reference does (float64 then float32).

#pragma once

#include "CoreMinimal.h"

struct LEARNEDOPPONENTS_API FLOSpecConfig
{
	int32 NRays = 24;
	double RayFovDeg = 360.0;
	double MaxRangeM = 4.0;
	double ArenaSizeM = 8.0;
	double WheelRadiusM = 0.05;
	double TrackWidthM = 0.30;
	double MaxWheelSpeed = 20.0;

	int32 ObsDim() const { return NRays + 5; }
	double VMax() const { return WheelRadiusM * MaxWheelSpeed; }
	double OmegaMax() const { return 2.0 * WheelRadiusM * MaxWheelSpeed / TrackWidthM; }
	double ArenaDiagonalM() const { return FMath::Sqrt(2.0) * ArenaSizeM; }
};

namespace LOObs
{
	// Ray angles in the robot frame, index order (index 0 straight back for a 360 degree fan).
	LEARNEDOPPONENTS_API TArray<double> RayAnglesRad(const FLOSpecConfig& Cfg);

	// Miss (negative), above max range, or non-finite -> max range; then divide and clip to [0, 1].
	LEARNEDOPPONENTS_API double NormalizeRay(double RangeM, double MaxRangeM);

	// Build the observation from spec-frame inputs. RangesM holds NRays raw distances (negative = miss).
	LEARNEDOPPONENTS_API bool BuildObservation(const FLOSpecConfig& Cfg, TConstArrayView<double> RangesM,
		double RobotX, double RobotY, double RobotYaw, double GoalX, double GoalY,
		double VelLinear, double VelAngular, TArray<float>& OutObs);

	// Unreal -> spec conversions.
	LEARNEDOPPONENTS_API FVector2D UnrealToSpecXY(double XCm, double YCm);
	LEARNEDOPPONENTS_API double UnrealToSpecYaw(double YawDeg);
	LEARNEDOPPONENTS_API double UnrealToSpecYawRate(double YawRateDegPerS);
	LEARNEDOPPONENTS_API double UnrealToSpecSpeed(double SpeedCmPerS);
	LEARNEDOPPONENTS_API double UnrealToSpecRange(double RangeCm);   // negative (miss) stays negative

	// Spec -> Unreal.
	LEARNEDOPPONENTS_API FVector SpecToUnrealLocation(double X, double Y, double ZCm);
	LEARNEDOPPONENTS_API double SpecToUnrealYawDeg(double YawRad);
	LEARNEDOPPONENTS_API double SpecRayAngleToUnrealYawOffsetDeg(double ThetaRad);

	// Build the observation from Unreal-frame inputs (positions in cm, yaw in degrees clockwise,
	// speed in cm/s along the actor's forward vector, yaw rate in deg/s, ranges in cm).
	LEARNEDOPPONENTS_API bool BuildObservationFromUnreal(const FLOSpecConfig& Cfg, TConstArrayView<double> RangesCm,
		const FVector& RobotCm, double RobotYawDeg, const FVector& GoalCm,
		double SpeedCmPerS, double YawRateDegPerS, TArray<float>& OutObs);
}
