# goal: given a NeRO like dataset, get an idea of the learnable parts of the env map

NeRO format dataset --> binary map (1) showing learnable parts of the env map (0) otherwise

data/ (in .gitignore) has egg_10_30 dataset created using single_obj render script in [gs-dataset](https://github.com/ahitagnied/gs-dataset)

## concept

a part of the environment map is learnable if light from that direction can be observed by the cameras - determined by whether rays from camera viewpoints intersect that spherical direction.

## plan

sample rays within each camera's field of view, convert ray directions to spherical coordinates (azimuth/elevation), and project onto an equirectangular environment map to mark visible regions.