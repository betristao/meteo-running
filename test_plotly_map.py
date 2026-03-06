import pandas as pd
import plotly.express as px
import random

lats = [38.7 + i*0.001 for i in range(100)]
lons = [-9.14 + i*0.001 for i in range(100)]
wind = ["Contra", "A favor", "Lateral"] * 34
wind = wind[:100]

df = pd.DataFrame({'lat': lats, 'lon': lons, 'wind': wind})
fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="wind",
                        color_discrete_map={"Contra": "red", "A favor": "green", "Lateral": "yellow"},
                        zoom=12, mapbox_style="carto-positron")
fig.write_image("test_map.png")
