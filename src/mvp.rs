use nalgebra_glm::identity;
use nalgebra_glm::TMat4;

#[derive(Debug, Clone)]
pub struct MVP {
    pub model: TMat4<f32>,
    pub view: TMat4<f32>,
    pub projection: TMat4<f32>,
}

impl MVP {
    pub fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            projection: identity(),
        }
    }
}
