#include "Race/LOArena.h"

#include "Components/BoxComponent.h"
#include "Components/CapsuleComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Obs/LOObsSpec.h"

namespace
{
	const TCHAR* CubePath = TEXT("/Engine/BasicShapes/Cube.Cube");
	const TCHAR* CylinderPath = TEXT("/Engine/BasicShapes/Cylinder.Cylinder");
	const TCHAR* PlanePath = TEXT("/Engine/BasicShapes/Plane.Plane");
	const FLinearColor FloorColor(0.30f, 0.30f, 0.30f);
	const FLinearColor WallColor(0.08f, 0.08f, 0.10f);
	const FLinearColor ObstacleColor(0.85f, 0.40f, 0.12f);
	constexpr double ParkedZ = -10000.0;
	constexpr double CapsuleExtraHalfHeightCm = 100.0;

	void SetStaticCollision(UPrimitiveComponent* C)
	{
		C->SetMobility(EComponentMobility::Movable);
		C->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
		C->SetCollisionObjectType(ECC_WorldStatic);
		C->SetCollisionResponseToAllChannels(ECR_Block);
	}
}

ALOArena::ALOArena()
{
	PrimaryActorTick.bCanEverTick = false;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

UMaterialInterface* ALOArena::ColorMaterial(UObject* Outer, const FLinearColor& Color)
{
	UMaterialInterface* Base = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
	if (!Base)
	{
		return nullptr;
	}
	UMaterialInstanceDynamic* Mat = UMaterialInstanceDynamic::Create(Base, Outer);
	Mat->SetVectorParameterValue(TEXT("Color"), Color);
	return Mat;
}

UStaticMeshComponent* ALOArena::AddMesh(const TCHAR* Path, const FLinearColor& Color)
{
	UStaticMeshComponent* Mesh = NewObject<UStaticMeshComponent>(this);
	Mesh->SetStaticMesh(LoadObject<UStaticMesh>(nullptr, Path));
	Mesh->SetMobility(EComponentMobility::Movable);
	Mesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	Mesh->SetMaterial(0, ColorMaterial(this, Color));
	Mesh->SetupAttachment(RootComponent);
	Mesh->RegisterComponent();
	AddInstanceComponent(Mesh);
	return Mesh;
}

void ALOArena::Build(const FLOTaskData& Task)
{
	TaskData = Task;
	const double HalfCm = Task.Spec.ArenaSizeM * 50.0;
	const double TCm = 5.0;                                // wall half-thickness: 0.1 m walls, as in MuJoCo
	const double WallHalfHCm = Task.WallHeightM * 50.0;

	UStaticMeshComponent* Floor = AddMesh(PlanePath, FloorColor);
	Floor->SetWorldLocation(FVector(0, 0, 0));
	Floor->SetWorldScale3D(FVector((HalfCm * 2 + 200) / 100.0, (HalfCm * 2 + 200) / 100.0, 1.0));

	const FVector WallCentres[4] = {FVector(0, HalfCm + TCm, WallHalfHCm), FVector(0, -HalfCm - TCm, WallHalfHCm),
		FVector(HalfCm + TCm, 0, WallHalfHCm), FVector(-HalfCm - TCm, 0, WallHalfHCm)};
	const FVector WallExtents[4] = {FVector(HalfCm + 2 * TCm, TCm, WallHalfHCm), FVector(HalfCm + 2 * TCm, TCm, WallHalfHCm),
		FVector(TCm, HalfCm, WallHalfHCm), FVector(TCm, HalfCm, WallHalfHCm)};
	for (int32 I = 0; I < 4; ++I)
	{
		UBoxComponent* Wall = NewObject<UBoxComponent>(this);
		Wall->SetupAttachment(RootComponent);
		SetStaticCollision(Wall);
		Wall->SetBoxExtent(WallExtents[I], false);
		Wall->SetWorldLocation(WallCentres[I]);
		Wall->RegisterComponent();
		AddInstanceComponent(Wall);
		UStaticMeshComponent* Visual = AddMesh(CubePath, WallColor);
		Visual->SetWorldLocation(WallCentres[I]);
		Visual->SetWorldScale3D(WallExtents[I] * 2.0 / 100.0);
	}

	for (int32 J = 0; J < Task.MaxObstacles; ++J)
	{
		UBoxComponent* Box = NewObject<UBoxComponent>(this);
		Box->SetupAttachment(RootComponent);
		SetStaticCollision(Box);
		Box->RegisterComponent();
		AddInstanceComponent(Box);
		BoxSlots.Add(Box);
		BoxVisuals.Add(AddMesh(CubePath, ObstacleColor));

		UCapsuleComponent* Cyl = NewObject<UCapsuleComponent>(this);
		Cyl->SetupAttachment(RootComponent);
		SetStaticCollision(Cyl);
		Cyl->RegisterComponent();
		AddInstanceComponent(Cyl);
		CylinderSlots.Add(Cyl);
		CylinderVisuals.Add(AddMesh(CylinderPath, ObstacleColor));
	}
	ApplyLayout(FLOLayout());
}

void ALOArena::ApplyLayout(const FLOLayout& Layout)
{
	const double HalfHCm = TaskData.ObstacleHeightM * 50.0;
	for (int32 J = 0; J < BoxSlots.Num(); ++J)
	{
		const FLOObstacle* Ob = Layout.Obstacles.IsValidIndex(J) ? &Layout.Obstacles[J] : nullptr;
		const bool bBox = Ob && Ob->Kind == ELOObstacleKind::Box;
		const bool bCyl = Ob && Ob->Kind == ELOObstacleKind::Cylinder;
		const FVector Centre = Ob ? LOObs::SpecToUnrealLocation(Ob->X, Ob->Y, HalfHCm) : FVector(0, 0, ParkedZ);
		const double SizeCm = Ob ? Ob->Size * 100.0 : 1.0;

		UBoxComponent* Box = BoxSlots[J];
		Box->SetBoxExtent(FVector(SizeCm / 2.0, SizeCm / 2.0, HalfHCm), false);
		Box->SetWorldLocation(bBox ? Centre : FVector(0, 0, ParkedZ));
		Box->SetCollisionEnabled(bBox ? ECollisionEnabled::QueryOnly : ECollisionEnabled::NoCollision);
		BoxVisuals[J]->SetWorldLocation(bBox ? Centre : FVector(0, 0, ParkedZ));
		BoxVisuals[J]->SetWorldScale3D(FVector(SizeCm / 100.0, SizeCm / 100.0, HalfHCm * 2.0 / 100.0));
		BoxVisuals[J]->SetVisibility(bBox);

		UCapsuleComponent* Cyl = CylinderSlots[J];
		const double R = SizeCm / 2.0;
		Cyl->SetCapsuleSize(R, R + CapsuleExtraHalfHeightCm, false);
		Cyl->SetWorldLocation(bCyl ? Centre : FVector(0, 0, ParkedZ));
		Cyl->SetCollisionEnabled(bCyl ? ECollisionEnabled::QueryOnly : ECollisionEnabled::NoCollision);
		CylinderVisuals[J]->SetWorldLocation(bCyl ? Centre : FVector(0, 0, ParkedZ));
		CylinderVisuals[J]->SetWorldScale3D(FVector(SizeCm / 100.0, SizeCm / 100.0, HalfHCm * 2.0 / 100.0));
		CylinderVisuals[J]->SetVisibility(bCyl);
	}
}
