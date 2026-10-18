// In-engine inference with Unreal's Neural Network Engine on the ONNX Runtime CPU backend.
//
// The exported ONNX file is the whole deterministic policy (normalization, MLP, clip), with
// input "obs" float32[1, n_rays + 5] and output "action" float32[1, 2]. The model data is
// created at runtime from the file bytes with UNNEModelData::Init, so no editor import is needed.

#pragma once

#include "CoreMinimal.h"
#include "NNEModelData.h"   // complete type: TStrongObjectPtr is built wherever FLOPolicy is copied
#include "UObject/StrongObjectPtr.h"

namespace UE::NNE
{
	class IModelCPU;
	class IModelInstanceCPU;
}

class LEARNEDOPPONENTS_API FLOPolicy
{
public:
	static constexpr const TCHAR* RuntimeName = TEXT("NNERuntimeORTCpu");

	bool Load(const FString& OnnxPath, int32 InObsDim, FString& OutError);
	bool IsLoaded() const { return Instance.IsValid(); }

	// One forward pass: Obs (ObsDim floats) -> OutAction (2 floats).
	bool Run(TConstArrayView<float> Obs, float OutAction[2]);

	const FString& Path() const { return ModelPath; }

private:
	TStrongObjectPtr<UNNEModelData> ModelData;
	TSharedPtr<UE::NNE::IModelCPU> Model;
	TSharedPtr<UE::NNE::IModelInstanceCPU> Instance;
	FString ModelPath;
	int32 ObsDim = 0;
};
