import xml.etree.ElementTree as ET
import math

def parse_gpx(xml_string):
    root = ET.fromstring(xml_string)
    # namespaces in gpx are usually like {http://www.topografix.com/GPX/1/1}
    # we can just ignore namespace by using a wildcard or stripping it
    points = []
    for trkpt in root.iter():
        if 'trkpt' in trkpt.tag:
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            points.append((lat, lon))
    return points

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

gpx_data = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Strava">
  <trk>
    <trkseg>
      <trkpt lat="38.736946" lon="-9.142685"></trkpt>
      <trkpt lat="38.737" lon="-9.142"></trkpt>
      <trkpt lat="38.74" lon="-9.14"></trkpt>
    </trkseg>
  </trk>
</gpx>
"""
pts = parse_gpx(gpx_data)
print(pts)
dist = 0
for i in range(1, len(pts)):
    d = haversine(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
    b = bearing(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
    dist += d
    print(f"To {i}: dist={d:.3f} km, accum={dist:.3f} km, bear={b:.1f}")
