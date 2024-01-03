use nalgebra_glm::{pi, vec3};

use crate::render_system::{self, lighting::directional::DirectionalLight, model::Model};

use super::Scene;

pub struct TeapotScene;

impl Scene for TeapotScene {
    fn init(&self) {}

    fn draw(&self, application: &mut render_system::RenderSystem) {
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

        application.add_geometry(&mut teapot);
        application.draw_ambient_light();
        application.draw_directional_light(&directional_light_red);
        application.draw_directional_light(&directional_light_green);
    }
}
