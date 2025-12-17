TO DO

Implement refraction and render the sphere in the default scene as a refractive object with index of refraction (IoR) n = 1.5. 
Refraction requires that you keep track of the relative index of refraction for each intersection with the refractive object. 
Use the sign of the dot product between the surface normal and the ray direction to find out whether a ray hit the surface from the inside or from the outside and store the corresponding relative index of refraction in the HitInfo struct