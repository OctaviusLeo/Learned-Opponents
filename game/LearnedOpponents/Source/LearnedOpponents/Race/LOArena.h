// Arena built from engine basic shapes: floor (visual only), four walls, and obstacle slots.
//
// Collision uses exact analytic shapes so the geometry matches MuJoCo: walls and boxes are
// UBoxComponents; round obstacles are vertical UCapsuleComponents whose straight section spans
// far above and below the ray/robot plane, so at that plane their cross-section is exactly the
// cylinder's circle. Visuals are basic-shape meshes with collision disabled.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Obs/LOGameData.h"
#include "LOArena.generated.h"

class UBoxComponent;
class UCapsuleComponent;
class UStaticMeshComponent;
class UMaterialInterface;

UCLASS()
class LEARNEDOPPONENTS_API ALOArena : public AActor
{
	GENERATED_BODY()

public:
	ALOArena();

	// Create floor, walls, and MaxObstacles box + capsule slots (all inactive).
	void Build(const FLOTaskData& Task);

	// Activate and size the slots for this layout; unused slots lose collision and visibility.
	void ApplyLayout(const FLOLayout& Layout);

	static UMaterialInterface* ColorMaterial(UObject* Outer, const FLinearColor& Color);

private:
	UStaticMeshComponent* AddMesh(const TCHAR* Path, const FLinearColor& Color);

	UPROPERTY()
	TArray<TObjectPtr<UBoxComponent>> BoxSlots;
	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> BoxVisuals;
	UPROPERTY()
	TArray<TObjectPtr<UCapsuleComponent>> CylinderSlots;
	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> CylinderVisuals;

	FLOTaskData TaskData;
};
