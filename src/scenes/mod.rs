use crate::render_system;

pub mod monkey;
pub mod teapot;

pub trait Scene {
    fn init(&self);
    fn draw(&self, application: &mut render_system::RenderSystem);
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

    pub fn active_scene(&self) -> &dyn Scene {
        &*self.scenes[self.active_scene_index as usize]
    }
}
