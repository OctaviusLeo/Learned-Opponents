#include "Inference/LONpy.h"

#include "Misc/FileHelper.h"

namespace LONpy
{
	bool LoadFloat32(const FString& Path, TArray<float>& OutData, TArray<int64>& OutShape, FString& OutError)
	{
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *Path))
		{
			OutError = FString::Printf(TEXT("cannot read %s"), *Path);
			return false;
		}
		static const uint8 Magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
		if (Bytes.Num() < 10 || FMemory::Memcmp(Bytes.GetData(), Magic, 6) != 0)
		{
			OutError = FString::Printf(TEXT("%s is not a .npy file"), *Path);
			return false;
		}
		const uint8 Major = Bytes[6];
		int64 HeaderLen = 0;
		int64 HeaderStart = 0;
		if (Major == 1)
		{
			HeaderLen = Bytes[8] | (Bytes[9] << 8);
			HeaderStart = 10;
		}
		else
		{
			HeaderLen = Bytes[8] | (Bytes[9] << 8) | (Bytes[10] << 16) | (int64(Bytes[11]) << 24);
			HeaderStart = 12;
		}
		const FString Header = FString::ConstructFromPtrSize(reinterpret_cast<const ANSICHAR*>(Bytes.GetData() + HeaderStart),
			static_cast<int32>(HeaderLen));
		if (!Header.Contains(TEXT("'descr': '<f4'")) || !Header.Contains(TEXT("'fortran_order': False")))
		{
			OutError = FString::Printf(TEXT("%s: expected little-endian float32 in C order, header %s"), *Path, *Header);
			return false;
		}
		const int32 Open = Header.Find(TEXT("("));
		const int32 Close = Header.Find(TEXT(")"));
		TArray<FString> Parts;
		Header.Mid(Open + 1, Close - Open - 1).ParseIntoArray(Parts, TEXT(","), true);
		OutShape.Reset();
		int64 Count = 1;
		for (FString& P : Parts)
		{
			P.TrimStartAndEndInline();
			const int64 D = FCString::Atoi64(*P);
			OutShape.Add(D);
			Count *= D;
		}
		const int64 DataStart = HeaderStart + HeaderLen;
		if (Bytes.Num() - DataStart != Count * int64(sizeof(float)))
		{
			OutError = FString::Printf(TEXT("%s: data size does not match shape"), *Path);
			return false;
		}
		OutData.SetNumUninitialized(static_cast<int32>(Count));
		FMemory::Memcpy(OutData.GetData(), Bytes.GetData() + DataStart, Count * sizeof(float));
		return true;
	}
}
