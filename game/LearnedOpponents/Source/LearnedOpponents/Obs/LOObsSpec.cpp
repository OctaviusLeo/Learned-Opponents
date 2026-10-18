#include "Obs/LOObsSpec.h"

namespace LOObs
{
	TArray<double> RayAnglesRad(const FLOSpecConfig& Cfg)
	{
		TArray<double> Out;
		Out.Reserve(Cfg.NRays);
		const double StepDeg = Cfg.RayFovDeg >= 360.0 ? Cfg.RayFovDeg / Cfg.NRays : Cfg.RayFovDeg / (Cfg.NRays - 1);
		for (int32 I = 0; I < Cfg.NRays; ++I)
		{
			Out.Add(FMath::DegreesToRadians(-Cfg.RayFovDeg / 2.0 + I * StepDeg));
		}
		return Out;
	}

	double NormalizeRay(double RangeM, double MaxRangeM)
	{
		if (!FMath::IsFinite(RangeM) || RangeM < 0.0 || RangeM > MaxRangeM)
		{
			RangeM = MaxRangeM;
		}
		return FMath::Clamp(RangeM / MaxRangeM, 0.0, 1.0);
	}

	bool BuildObservation(const FLOSpecConfig& Cfg, TConstArrayView<double> RangesM,
		double RobotX, double RobotY, double RobotYaw, double GoalX, double GoalY,
		double VelLinear, double VelAngular, TArray<float>& OutObs)
	{
		if (RangesM.Num() != Cfg.NRays)
		{
			return false;
		}
		OutObs.SetNumUninitialized(Cfg.ObsDim());
		for (int32 I = 0; I < Cfg.NRays; ++I)
		{
			OutObs[I] = static_cast<float>(NormalizeRay(RangesM[I], Cfg.MaxRangeM));
		}
		// Rotate the world-frame goal offset into the robot frame (rotation by -yaw).
		const double Dx = GoalX - RobotX;
		const double Dy = GoalY - RobotY;
		const double C = FMath::Cos(RobotYaw);
		const double S = FMath::Sin(RobotYaw);
		const double Lx = C * Dx + S * Dy;
		const double Ly = -S * Dx + C * Dy;
		const double Dist = FMath::Sqrt(Lx * Lx + Ly * Ly);
		const double Gx = Dist > 1e-9 ? Lx / Dist : 0.0;
		const double Gy = Dist > 1e-9 ? Ly / Dist : 0.0;
		const int32 N = Cfg.NRays;
		OutObs[N + 0] = static_cast<float>(FMath::Clamp(Gx, -1.0, 1.0));
		OutObs[N + 1] = static_cast<float>(FMath::Clamp(Gy, -1.0, 1.0));
		OutObs[N + 2] = static_cast<float>(FMath::Clamp(Dist / Cfg.ArenaDiagonalM(), 0.0, 1.0));
		OutObs[N + 3] = static_cast<float>(FMath::Clamp(VelLinear / Cfg.VMax(), -1.0, 1.0));
		OutObs[N + 4] = static_cast<float>(FMath::Clamp(VelAngular / Cfg.OmegaMax(), -1.0, 1.0));
		return true;
	}

	FVector2D UnrealToSpecXY(double XCm, double YCm) { return FVector2D(XCm / 100.0, -YCm / 100.0); }
	double UnrealToSpecYaw(double YawDeg) { return -FMath::DegreesToRadians(YawDeg); }
	double UnrealToSpecYawRate(double YawRateDegPerS) { return -FMath::DegreesToRadians(YawRateDegPerS); }
	double UnrealToSpecSpeed(double SpeedCmPerS) { return SpeedCmPerS / 100.0; }
	double UnrealToSpecRange(double RangeCm) { return RangeCm < 0.0 ? RangeCm : RangeCm / 100.0; }

	FVector SpecToUnrealLocation(double X, double Y, double ZCm) { return FVector(X * 100.0, -Y * 100.0, ZCm); }
	double SpecToUnrealYawDeg(double YawRad) { return -FMath::RadiansToDegrees(YawRad); }
	double SpecRayAngleToUnrealYawOffsetDeg(double ThetaRad) { return -FMath::RadiansToDegrees(ThetaRad); }

	bool BuildObservationFromUnreal(const FLOSpecConfig& Cfg, TConstArrayView<double> RangesCm,
		const FVector& RobotCm, double RobotYawDeg, const FVector& GoalCm,
		double SpeedCmPerS, double YawRateDegPerS, TArray<float>& OutObs)
	{
		TArray<double> RangesM;
		RangesM.Reserve(RangesCm.Num());
		for (const double R : RangesCm)
		{
			RangesM.Add(UnrealToSpecRange(R));
		}
		const FVector2D Robot = UnrealToSpecXY(RobotCm.X, RobotCm.Y);
		const FVector2D Goal = UnrealToSpecXY(GoalCm.X, GoalCm.Y);
		return BuildObservation(Cfg, RangesM, Robot.X, Robot.Y, UnrealToSpecYaw(RobotYawDeg), Goal.X, Goal.Y,
			UnrealToSpecSpeed(SpeedCmPerS), UnrealToSpecYawRate(YawRateDegPerS), OutObs);
	}
}
