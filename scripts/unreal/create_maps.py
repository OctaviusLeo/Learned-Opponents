"""Create the two empty maps headless with Unreal's editor Python (task 15.1 / 16.2).

Run through scripts/create_maps.ps1. The maps stay empty: the game mode builds the arena,
robots, camera, and light from code at BeginPlay, so no level is edited by hand.
"""

import unreal

for name in ("RaceLevel", "TransferArena"):
    path = f"/Game/Maps/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.log(f"LO maps: {path} already exists")
        continue
    world = unreal.EditorLoadingAndSavingUtils.new_blank_map(False)
    saved = unreal.EditorLoadingAndSavingUtils.save_map(world, path)
    unreal.log(f"LO maps: created {path}: {saved}")
