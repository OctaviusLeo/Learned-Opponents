// Minimal reader for the float32, C-order .npy files numpy.save writes (parity fixtures).

#pragma once

#include "CoreMinimal.h"

namespace LONpy
{
	// Reads a little-endian float32 array; OutShape receives the dimensions.
	LEARNEDOPPONENTS_API bool LoadFloat32(const FString& Path, TArray<float>& OutData, TArray<int64>& OutShape, FString& OutError);
}
