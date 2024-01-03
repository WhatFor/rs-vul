use crate::render_system;

pub mod monkey;
pub mod teapot;

pub trait Scene {
    fn draw(&mut self, application: &mut render_system::RenderSystem);
}

pub struct SceneManager {
    scenes: Vec<Box<dyn Scene>>,
    active_scene_index: u32,
}

impl SceneManager {
    pub fn new() -> Self {
        SceneManager {
            active_scene_index: 0,
            scenes: Vec::new(),
        }
    }

    pub fn add_scene(&mut self, scene: Box<dyn Scene>) {
        self.scenes.push(scene);
    }

    pub fn set_active(&mut self, index: u32) {
        self.active_scene_index = index;
    }

    pub fn switch_scene_by_key(&mut self, input: winit::event::KeyboardInput) {
        if input.state != winit::event::ElementState::Pressed {
            return;
        }

        match input.virtual_keycode {
            Some(key) => {
                let key_number = key as u32;
                if key_number < self.scenes.len() as u32 && key_number != self.active_scene_index {
                    self.set_active(key_number);
                }
            }
            None => {}
        }
    }

    pub fn active_scene(&mut self) -> &mut dyn Scene {
        &mut *self.scenes[self.active_scene_index as usize]
    }
}
