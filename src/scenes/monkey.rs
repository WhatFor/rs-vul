use nalgebra_glm::{pi, vec3};

use crate::render_system::{self, lighting::directional::DirectionalLight, model::Model};

use super::Scene;

pub struct MonkeyScene {
    suzanne: Model,
    light_1: DirectionalLight,
    light_2: DirectionalLight,
}

impl MonkeyScene {
    pub fn new() -> MonkeyScene {
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

        MonkeyScene {
            suzanne,
            light_1: directional_light_red,
            light_2: directional_light_green,
        }
    }
}

impl Scene for MonkeyScene {
    fn draw(&mut self, application: &mut render_system::RenderSystem) {
        application.add_geometry(&mut self.suzanne);
        application.draw_ambient_light();
        application.draw_directional_light(&self.light_1);
        application.draw_directional_light(&self.light_2);
    }
}
