#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "LOHUD.generated.h"

UCLASS()
class LEARNEDOPPONENTS_API ALOHUD : public AHUD
{
	GENERATED_BODY()

public:
	virtual void DrawHUD() override;
};
