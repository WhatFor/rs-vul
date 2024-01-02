use nalgebra_glm::{vec3, TVec3};

#[derive(Default, Debug, Clone)]
pub struct DirectionalLight {
    pub position: [f32; 4],
    pub color: [f32; 3],
}

impl DirectionalLight {
    pub fn get_position(&self) -> TVec3<f32> {
        vec3(self.position[0], self.position[1], self.position[2])
    }
}
