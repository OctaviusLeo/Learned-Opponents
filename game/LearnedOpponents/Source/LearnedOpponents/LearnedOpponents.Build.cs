using UnrealBuildTool;

public class LearnedOpponents : ModuleRules
{
	public LearnedOpponents(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		// Sources live in subfolders (Obs/, Inference/, Race/, Transfer/, Tests/) and include each
		// other as "Obs/LOObsSpec.h"; the module root is not on the include path by default.
		PublicIncludePaths.Add(ModuleDirectory);

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"NNE",
			"Json",
			"JsonUtilities"
		});
	}
}
