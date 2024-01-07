use nalgebra_glm::{pi, vec3};

use crate::render_system::{lighting::directional::DirectionalLight, model::Model, RenderSystem};

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
    fn draw(&mut self, render_system: &mut RenderSystem) {
        //fn draw(&mut self, context: &mut SceneContext) {
        // find the magnitude of 100hz
        let fft_data = render_system.fft_container.read_fft(5);
        let mut magnitude = match fft_data.len() {
            0 => 0.0,
            _ => fft_data[100 / 44100],
        };

        magnitude = map_range(magnitude, 0.0, 1.0, 1.0, 1.5);

        log::info!("magnitude: {}", magnitude);

        // scale the monkey
        self.suzanne.scale(magnitude);

        render_system.add_geometry(&mut self.suzanne);
        render_system.draw_ambient_light();
        render_system.draw_directional_light(&self.light_1);
        render_system.draw_directional_light(&self.light_2);
    }
}

fn map_range(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    (value.min(from_max) - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
}
