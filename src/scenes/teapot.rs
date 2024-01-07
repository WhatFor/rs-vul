use nalgebra_glm::vec3;

use crate::render_system::{lighting::directional::DirectionalLight, model::Model, RenderSystem};

use super::Scene;

pub struct TeapotScene {
    teapot: Model,
    light_1: DirectionalLight,
    light_2: DirectionalLight,
}

impl TeapotScene {
    pub fn new() -> Self {
        let mut teapot = Model::new("resources/models/teapot.obj")
            .invert_winding_order(true)
            .build();

        teapot.translate(vec3(0.0, 0.0, -8.0));

        let directional_light_red = DirectionalLight {
            color: [1.0, 0.0, 0.0],
            position: [-4.0, -4.0, 0.0, -2.0],
        };

        let directional_light_green = DirectionalLight {
            color: [0.0, 1.0, 0.0],
            position: [4.0, -4.0, 0.0, -2.0],
        };

        TeapotScene {
            teapot,
            light_1: directional_light_red,
            light_2: directional_light_green,
        }
    }
}

impl Scene for TeapotScene {
    fn draw(&mut self, render_system: &mut RenderSystem) {
        render_system.add_geometry(&mut self.teapot);
        render_system.draw_ambient_light();
        render_system.draw_directional_light(&self.light_1);
        render_system.draw_directional_light(&self.light_2);
    }
}
