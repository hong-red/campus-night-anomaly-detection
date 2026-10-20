#!/usr/bin/env python3
import os as _os, sys as _sys
_cur = _os.path.dirname(_os.path.abspath(__file__))
for _i in range(6):
    if _os.path.exists(_os.path.join(_cur, "_paths.py")):
        _sys.path.insert(0, _cur)
        break
    _cur = _os.path.dirname(_cur)
import _paths as _PKGP
# -*- coding: utf-8 -*-
"""用 OSM Nominatim 抓取大学校园边界多边形 -> GeoJSON(与 geofence_tsinghua.json 同格式)。

用法:
    python fetch_geofence.py "<地名>" <输出.json>

例:
    python fetch_geofence.py "Stanford University" geofence_stanford.json
    python fetch_geofence.py "ETH Zürich" geofence_eth.json

说明: Nominatim 返回 geojson 多边形; 若为 MultiPolygon, 取面积最大的外环。
"""
import json, sys, urllib.request, urllib.parse

NOMINATIM = "https://nominatim.openstreetmap.org/search"


def fetch(query):
    params = urllib.parse.urlencode({
        "q": query, "format": "jsonv2", "polygon_geojson": "1", "limit": "1",
    })
    url = f"{NOMINATIM}?{params}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "campus-geofence/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def area_deg(ring):
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return (max(lats) - min(lats)) * (max(lons) - min(lons))


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    query, out = sys.argv[1], sys.argv[2]
    a = fetch(query)
    if not a:
        print("无结果:", query); sys.exit(1)
    e = a[0]
    g = e.get("geojson") or {}
    gtype, coords = g.get("type"), g.get("coordinates")
    if gtype == "Polygon":
        rings = coords
    elif gtype == "MultiPolygon":
        rings = [ring for poly in coords for ring in poly]
    else:
        print("返回类型不支持:", gtype, "display_name=", e.get("display_name")); sys.exit(1)
    best = max(rings, key=area_deg)
    out_obj = {
        "source": "OSM Nominatim",
        "name": e.get("display_name"),
        "osm_type": e.get("osm_type"),
        "osm_id": e.get("osm_id"),
        "type": "Polygon",
        "coordinates": [best],   # GeoJSON [lon, lat]
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print("已保存:", out)
    print("name:", e.get("display_name"))
    print("bbox:", e.get("boundingbox"))
    print("顶点数:", len(best))


if __name__ == "__main__":
    main()
