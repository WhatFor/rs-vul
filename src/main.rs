use fast_log::Config;
use jack::ClientOptions;
use rustfft::{num_complex::Complex, FftPlanner};
use scenes::{monkey::MonkeyScene, teapot::TeapotScene, SceneManager};

use log;
use vulkano::sync::GpuFuture;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::midi::{MidiEvent, MidiEventType};

pub mod audio;
pub mod midi;
pub mod render_system;
pub mod scenes;
pub mod shaders;

const LIVE_AUDIO_OUT_L: &str = "Ableton Live 11 Suite:out1";
const LIVE_AUDIO_OUT_R: &str = "Ableton Live 11 Suite:out2";
const AUDIO_FFT_CHUNK_SIZE: usize = 512;

fn main() {
    // Configure Logger
    fast_log::init(Config::new().console().chan_len(Some(100000))).unwrap();

    // Build Channels to send audio data from the JACK thread to the event_loop
    let (av_sender, av_receiver) = std::sync::mpsc::channel();
    let (midi_sender, midi_receiver) = std::sync::mpsc::channel();

    // Init Jack
    let (jack, status) = jack::Client::new("rs", ClientOptions::NO_START_SERVER).unwrap();
    log::info!("JACK client '{}', Status: {:?}", jack.name(), status);

    let in_port_audio_l = jack.register_port("audio_in_l", jack::AudioIn).unwrap();
    let in_port_audio_r = jack.register_port("audio_in_r", jack::AudioIn).unwrap();

    let live_audio_out_l = jack
        .port_by_name(LIVE_AUDIO_OUT_L)
        .expect("Live audio out L not found. Try reselecting audio device in Ableton.");
    let live_audio_out_r = jack
        .port_by_name(LIVE_AUDIO_OUT_R)
        .expect("Live audio out R not found. Try reselecting audio device in Ableton.");

    let midi_in_1 = jack.register_port("midi_in_1", jack::MidiIn).unwrap();
    let live_midi_loop_1 = jack.port_by_name("system_midi:capture_1").unwrap();

    // Init FFT
    let mut audio_1_fft_planner = FftPlanner::new();
    let audio_1_fft = audio_1_fft_planner.plan_fft_forward(AUDIO_FFT_CHUNK_SIZE);
    let mut audio_1_fft_buffer = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        AUDIO_FFT_CHUNK_SIZE
    ];
    let mut audio_1_fft_output = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        AUDIO_FFT_CHUNK_SIZE
    ];

    let mut is_jack_connected: bool = false;

    let process_events = move |_client: &jack::Client, ps: &jack::ProcessScope| -> jack::Control {
        if !is_jack_connected {
            // can only connect ports after the client is activated
            _client
                .connect_ports(&live_audio_out_l, &in_port_audio_l)
                .unwrap();
            _client
                .connect_ports(&live_audio_out_r, &in_port_audio_r)
                .unwrap();
            _client
                .connect_ports(&live_midi_loop_1, &midi_in_1)
                .unwrap();
            is_jack_connected = true;
        }

        let midi_in = midi_in_1.iter(ps);

        for raw_midi_event in midi_in {
            match raw_midi_event {
                jack::RawMidi { bytes, time: _ } => {
                    if bytes.len() >= 3 {
                        let status = bytes[0];
                        let channel = status - if status >= 144 { 143 } else { 127 };

                        let event_type = if status >= 144 && status <= 159 {
                            MidiEventType::NoteOn
                        } else {
                            MidiEventType::NoteOff
                        };

                        if channel > 16 {
                            // For now, ignore MIDI events on channels > 16
                            // as these are non-note events.
                            continue;
                        }

                        let velocity = if event_type == MidiEventType::NoteOn {
                            Some(bytes[2])
                        } else {
                            None
                        };

                        let midi_event = MidiEvent::new(event_type, channel, bytes[1], velocity);

                        midi_sender
                            .send(midi_event)
                            .expect("Unable to send MIDI event");
                    }
                }
            }
        }

        let audio_in_l = in_port_audio_l.as_slice(ps);

        for (i, sample) in audio_in_l.iter().enumerate() {
            audio_1_fft_buffer[i] = Complex {
                re: *sample,
                im: 0.0f32,
            };
        }

        audio_1_fft.process_with_scratch(&mut audio_1_fft_buffer, &mut audio_1_fft_output);

        // get the frequency analysis magnitudes
        let mut audio_1_fft_magnitudes = vec![0.0f32; AUDIO_FFT_CHUNK_SIZE];
        for (i, complex) in audio_1_fft_output.iter().enumerate() {
            audio_1_fft_magnitudes[i] = (complex.re.powi(2) + complex.im.powi(2)).sqrt();
        }

        av_sender
            .send(audio_1_fft_magnitudes)
            .expect("Unable to send audio data");

        jack::Control::Continue
    };

    // Start JACK
    let process = jack::ClosureProcessHandler::new(process_events);
    let _client = jack.activate_async({}, process).unwrap();

    // Init Render System
    let event_loop = EventLoop::new();
    let mut scene_man = SceneManager::new();
    let (mut sys, mut previous_frame_end) = render_system::RenderSystem::new(&event_loop);

    // Wire up Channels
    sys.set_midi_channel(midi_receiver);
    sys.set_audio_channel(av_receiver);

    // Add Scenes
    scene_man.add_scene(Box::new(MonkeyScene::new()));
    scene_man.add_scene(Box::new(TeapotScene::new()));

    // Start Rendering
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
            sys.recreate_swapchain();
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { input, .. },
            ..
        } => {
            scene_man.switch_scene_by_key(input);
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
            sys.start_frame();

            let active_scene = scene_man.active_scene();
            active_scene.draw(&mut sys);

            sys.finish_frame(&mut previous_frame_end);
        }
        _ => (),
    });
}
