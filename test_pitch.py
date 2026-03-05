from statsbombpy import sb
from src.tools.pitch_visualizations import PitchVisualizer

events = sb.events(match_id=3869685)
messi = events[events["player"] == "Lionel Andrés Messi Cuccittini"]
print(f"Found {len(messi)} events for Messi")

viz = PitchVisualizer()
viz.generate_shot_map(messi, "Messi")
viz.generate_heatmap(messi, "Messi", ["Pass", "Carry"])
viz.generate_pass_map(messi, "Messi")

print("Done! Check data/charts/")
