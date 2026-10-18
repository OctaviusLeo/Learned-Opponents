#include "Race/LOHUD.h"

#include "Engine/Canvas.h"
#include "EngineUtils.h"
#include "Race/LORaceManager.h"

void ALOHUD::DrawHUD()
{
	Super::DrawHUD();
	if (!Canvas)
	{
		return;
	}
	for (TActorIterator<ALORaceManager> It(GetWorld()); It; ++It)
	{
		float Y = 16.f;
		for (const FString& Line : It->StatusLines())
		{
			DrawText(Line, FLinearColor::White, 16.f, Y, nullptr, 1.4f);
			Y += 26.f;
		}
		break;
	}
}
