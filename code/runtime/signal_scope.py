"""Associate signalized lane connectors with the assigned route movements.

Connector polygons overlap across opposing/crossing movements at junctions.
Intersection with a different movement's polygon is not a red-light entry.
This planner follows its assigned roadblock sequence; off-route motion is not
an alternative legal maneuver supported by this signal association.
"""
def controls_route(connector,route_ids):
    ids={str(x) for x in route_ids}
    if not ids:
        # Missing route context must not silently remove traffic constraints.
        return True
    return str(connector.get_roadblock_id()) in ids
