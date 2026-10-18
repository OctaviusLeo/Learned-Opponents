using UnrealBuildTool;

public class LearnedOpponentsTarget : TargetRules
{
	public LearnedOpponentsTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("LearnedOpponents");
	}
}
