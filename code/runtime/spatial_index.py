"""C/GEOS STRtree queries; preserve native dataframe row ordering exactly."""
import numpy as np
import types
from shapely.strtree import STRtree

def indexed_proximity(self,patch,layer):
 if layer not in self._planning_spatial_indexes:
  df=self._get_vector_map_layer(layer)
  self._planning_spatial_indexes[layer]=(STRtree(df['geometry'].values),df['fid'].to_numpy(copy=True))
 tree,ids=self._planning_spatial_indexes[layer]
 roi=getattr(self,'_planning_local_roi',None)
 if roi is not None:patch=patch.intersection(roi)
 rows=np.sort(tree.query(patch,predicate='intersects'))
 return [self.get_map_object(ids[i],layer) for i in rows]

def install(map_api):
 if hasattr(map_api,'_planning_spatial_indexes'):return
 map_api._planning_spatial_indexes={}
 map_api._get_proximity_map_object=types.MethodType(indexed_proximity,map_api)

def update_roi(map_api,ego,half_extent=150.):
 from shapely.geometry import Polygon
 c,s=np.cos(ego.heading),np.sin(ego.heading)
 corners=np.array([[-half_extent,-half_extent],[half_extent,-half_extent],[half_extent,half_extent],[-half_extent,half_extent]])
 map_api._planning_local_roi=Polygon(corners @ np.array([[c,s],[-s,c]]) + [ego.x,ego.y])
