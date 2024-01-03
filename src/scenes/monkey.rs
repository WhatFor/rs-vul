use nalgebra_glm::{pi, vec3};

use crate::render_system::{self, lighting::directional::DirectionalLight, model::Model};

use super::Scene;

pub struct MonkeyScene;

impl Scene for MonkeyScene {
    fn init(&self) {}

    fn draw(&self, application: &mut render_system::RenderSystem) {
        let mut suzanne = Model::new("resources/models/suzanne.obj")
            .invert_winding_order(true)
            .build();

        suzanne.translate(vec3(0.0, 0.0, -3.0));
        suzanne.rotate(pi(), vec3(0.0, 1.0, 0.0));

        let directional_light_red = DirectionalLight {
            color: [1.0, 0.0, 0.0],
            position: [-4.0, -4.0, 0.0, -2.0],
        };

        let directional_light_green = DirectionalLight {
            color: [0.0, 1.0, 0.0],
            position: [4.0, -4.0, 0.0, -2.0],
        };

        application.add_geometry(&mut suzanne);
        application.draw_ambient_light();
        application.draw_directional_light(&directional_light_red);
        application.draw_directional_light(&directional_light_green);
    }
}
