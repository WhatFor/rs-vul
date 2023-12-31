pub mod system;

fn main() {
    let application = system::RenderSystem::new();
    application.run();
}
