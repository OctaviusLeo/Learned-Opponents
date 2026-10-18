using UnrealBuildTool;

public class LearnedOpponentsEditorTarget : TargetRules
{
	public LearnedOpponentsEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("LearnedOpponents");
	}
}
