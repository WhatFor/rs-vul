use scenes::{monkey::MonkeyScene, teapot::TeapotScene, SceneManager};

use vulkano::sync::GpuFuture;

use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

pub mod render_system;
pub mod scenes;
pub mod shaders;

fn main() {
    let event_loop = EventLoop::new();
    let mut scene_man = SceneManager::new();

    scene_man.add_scene(Box::new(MonkeyScene {}));
    scene_man.add_scene(Box::new(TeapotScene {}));

    let (mut application, mut previous_frame_end) = render_system::RenderSystem::new(&event_loop);

    {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                application.recreate_swapchain();
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if input.state != winit::event::ElementState::Pressed {
                    return;
                }

                match input.virtual_keycode {
                    Some(key) => match key {
                        VirtualKeyCode::Key1 => {
                            scene_man.set_active(0);
                        }
                        VirtualKeyCode::Key2 => {
                            scene_man.set_active(1);
                        }
                        _ => {}
                    },
                    None => {}
                }
            }
            Event::RedrawEventsCleared => {
                previous_frame_end
                    .as_mut()
                    .take()
                    .unwrap()
                    .cleanup_finished();

                // let elapsed = rotation_start.elapsed().as_secs() as f32
                //     + rotation_start.elapsed().subsec_nanos() as f32 / 1_000_000_000.0;

                // let elapsed_as_radians = elapsed * 30.0 * (pi::<f32>() / 180.0);

                // Spinning Light
                // let orbit_radius = 2.0;
                // let x: f32 = orbit_radius * elapsed_as_radians.cos();
                // let z: f32 = -3.0 + (orbit_radius * elapsed_as_radians.sin());
                // let spot_light = DirectionalLight {
                //     color: [1.0, 0.9, 0.9],
                //     position: [x, 0.0, z, 1.0],
                // };

                // Translate
                // suzanne.zero_rotation();
                // suzanne.rotate(elapsed_as_radians, vec3(0.0, 0.0, 1.0));
                // suzanne.rotate(elapsed_as_radians, vec3(0.0, 1.0, 0.0));
                // suzanne.rotate(elapsed_as_radians, vec3(1.0, 0.0, 0.0));
                // suzanne.rotate(elapsed_as_radians * 50.0, vec3(0.0, 0.0, 1.0));
                // suzanne.rotate(elapsed_as_radians * 30.0, vec3(0.0, 1.0, 0.0));
                // suzanne.rotate(elapsed_as_radians * 20.0, vec3(1.0, 0.0, 0.0));

                // // Translate
                // teapot.zero_rotation();
                // teapot.rotate(elapsed_as_radians * 50.0, vec3(0.0, 0.0, 1.0));
                // teapot.rotate(elapsed_as_radians * 30.0, vec3(0.0, 1.0, 0.0));
                // teapot.rotate(elapsed_as_radians * 20.0, vec3(1.0, 0.0, 0.0));

                // Draw!
                application.start_frame();

                // application.add_geometry(&mut suzanne);
                // application.add_geometry(&mut teapot);
                // application.draw_ambient_light();
                // application.draw_directional_light(&directional_light_red);
                // application.draw_directional_light(&directional_light_green);

                // application.draw_directional_light(&spot_light);
                //application.draw_light_object(&spot_light);

                let active_scene = scene_man.active_scene();
                active_scene.draw(&mut application);

                application.finish_frame(&mut previous_frame_end);
            }
            _ => (),
        });
    }
}
