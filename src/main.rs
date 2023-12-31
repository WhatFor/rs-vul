pub mod render_system;
pub mod shaders;

fn main() {
    let application = render_system::RenderSystem::new();
    application.run();
}
