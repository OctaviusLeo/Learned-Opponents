using UnrealBuildTool;

public class LearnedOpponents : ModuleRules
{
	public LearnedOpponents(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

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
