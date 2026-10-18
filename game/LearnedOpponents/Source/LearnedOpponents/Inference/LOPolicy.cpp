#include "Inference/LOPolicy.h"

#include "Misc/FileHelper.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNEStatus.h"

bool FLOPolicy::Load(const FString& OnnxPath, int32 InObsDim, FString& OutError)
{
	ModelPath = OnnxPath;
	ObsDim = InObsDim;
	TArray<uint8> Bytes;
	if (!FFileHelper::LoadFileToArray(Bytes, *OnnxPath))
	{
		OutError = FString::Printf(TEXT("cannot read %s"), *OnnxPath);
		return false;
	}
	ModelData = TStrongObjectPtr<UNNEModelData>(NewObject<UNNEModelData>());
	ModelData->Init(TEXT("onnx"), Bytes);

	TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(RuntimeName);
	if (!Runtime.IsValid())
	{
		OutError = FString::Printf(TEXT("NNE runtime %s is not available (is the NNERuntimeORT plugin enabled?)"), RuntimeName);
		return false;
	}
	if (Runtime->CanCreateModelCPU(ModelData.Get()) != UE::NNE::EResultStatus::Ok)
	{
		OutError = FString::Printf(TEXT("%s cannot create a CPU model from %s"), RuntimeName, *OnnxPath);
		return false;
	}
	Model = Runtime->CreateModelCPU(ModelData.Get());
	if (!Model.IsValid())
	{
		OutError = TEXT("CreateModelCPU failed");
		return false;
	}
	Instance = Model->CreateModelInstanceCPU();
	if (!Instance.IsValid())
	{
		OutError = TEXT("CreateModelInstanceCPU failed");
		return false;
	}
	const uint32 Dims[2] = {1u, static_cast<uint32>(ObsDim)};
	const UE::NNE::FTensorShape Shape = UE::NNE::FTensorShape::Make(Dims);
	if (Instance->SetInputTensorShapes({Shape}) != UE::NNE::EResultStatus::Ok)
	{
		OutError = TEXT("SetInputTensorShapes([1, obs_dim]) failed");
		Instance.Reset();
		return false;
	}
	return true;
}

bool FLOPolicy::Run(TConstArrayView<float> Obs, float OutAction[2])
{
	if (!Instance.IsValid() || Obs.Num() != ObsDim)
	{
		return false;
	}
	UE::NNE::FTensorBindingCPU Input{const_cast<float*>(Obs.GetData()), static_cast<uint64>(ObsDim * sizeof(float))};
	UE::NNE::FTensorBindingCPU Output{OutAction, static_cast<uint64>(2 * sizeof(float))};
	return Instance->RunSync({Input}, {Output}) == UE::NNE::EResultStatus::Ok;
}
