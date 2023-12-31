pub mod lighting;
pub mod model;
pub mod obj_loader;
pub mod shaders;

use nalgebra_glm::identity;
use nalgebra_glm::TMat4;

#[derive(Debug, Clone)]
pub struct VP {
    pub view: TMat4<f32>,
    pub projection: TMat4<f32>,
}

impl VP {
    pub fn new() -> VP {
        VP {
            view: identity(),
            projection: identity(),
        }
    }
}
