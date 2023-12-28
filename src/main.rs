use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage,
    },
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    device::DeviceExtensions,
    image::ImageUsage,
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::graphics::viewport::Viewport,
    swapchain::{
        self, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, future::FenceSignalFuture, FlushError, GpuFuture},
};

use rs_vul::{
    obj,
    shaders::{gl_frag, gl_vert},
    vertex::Vert,
    vulkan::{
        create_logical_device, gen_framebuffers, get_command_buffers, get_instance, get_pipeline,
        get_render_pass, select_physical_device,
    },
};

const TITLE: &str = "RS VUL";
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct MainApplication {}

impl MainApplication {
    pub fn init() {
        let instance = get_instance();
        let event_loop = EventLoop::new();

        let surface = WindowBuilder::new()
            .with_title(TITLE)
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let window = surface
            .object()
            .unwrap()
            .clone()
            .downcast::<Window>()
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface, &device_extensions);

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) =
            create_logical_device(&physical_device, device_extensions, queue_family_index);

        let queue = queues.next().unwrap();

        let capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("Failed to get surface caps");

        let dimensions = window.inner_size();

        let composite_alpha = capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap();

        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        println!("Building Swapchain. Format: {:?}", image_format);

        let (mut swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: capabilities.min_image_count + 1, // how many buffers to use
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();

        println!("Building Memory Allocator...");
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Create shaders
        let vs = gl_vert::load(device.clone()).expect("Failed to compile VS");
        let fs = gl_frag::load(device.clone()).expect("Failed to compile FS");
        println!("Built shaders.");

        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        println!("Loading model...");
        let obj_data = obj::load_model("resources/models/duck.obj");

        println!("Model loaded. Transforming to vertex buffer...");
        let vertex_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            obj_data.verticies,
        )
        .unwrap();

        println!("Transforming to normals buffer...");
        let normals_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vec![0], // TODO: Normals!
        )
        .unwrap();

        println!("Transforming to index buffer...");
        let index_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            obj_data.indicies,
        )
        .unwrap();

        println!("Transforming to uniform buffer...");
        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        println!("Getting render pass...");
        let render_pass = get_render_pass(&device, &swapchain);
        let framebuffers = gen_framebuffers(&images, &render_pass);

        println!("Getting viewport...");
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: window.inner_size().into(),
            depth_range: 0.0..1.0,
        };

        println!("Getting pipeline...");
        let pipeline = get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        );

        println!("Create command buffers...");
        let mut command_buffers = get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &framebuffers,
            &vertex_buffer,
        );
        println!("Command buffers created");

        let mut window_resized = false;
        let mut recreate_swapchain = false;

        let frames_in_flight = images.len();

        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;

        event_loop.run(move |event, _, control_flow| {
            match event {
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
                    window_resized = true;
                }
                Event::MainEventsCleared => {}
                Event::RedrawEventsCleared => {
                    if recreate_swapchain {
                        recreate_swapchain = false;

                        let new_dimensions = window.inner_size();

                        let (new_swapchain, new_images) =
                            match swapchain.recreate(SwapchainCreateInfo {
                                image_extent: new_dimensions.into(),
                                ..swapchain.create_info()
                            }) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                                    return
                                }
                                Err(e) => panic!("Failed to recreate swapchain: {e}"),
                            };

                        swapchain = new_swapchain;
                        let new_framebuffers = gen_framebuffers(&new_images, &render_pass);

                        if window_resized {
                            window_resized = false;

                            viewport.dimensions = new_dimensions.into();

                            let new_pipeline = get_pipeline(
                                device.clone(),
                                vs.clone(),
                                fs.clone(),
                                render_pass.clone(),
                                viewport.clone(),
                            );

                            command_buffers = get_command_buffers(
                                &command_buffer_allocator,
                                &queue,
                                &new_pipeline,
                                &new_framebuffers,
                                &vertex_buffer,
                            );
                        }
                    }

                    let (image_i, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {e}"),
                        };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    // todo; need to understand below code

                    // wait for the fence related to this image to finish (normally this would be the oldest fence)
                    if let Some(image_fence) = &fences[image_i as usize] {
                        image_fence.wait(None).unwrap();
                    }

                    let previous_future = match fences[previous_fence_i as usize].clone() {
                        // Create a NowFuture
                        None => {
                            let mut now = sync::now(device.clone());
                            now.cleanup_finished();

                            now.boxed()
                        }
                        // Use the existing FenceSignalFuture
                        Some(fence) => fence.boxed(),
                    };

                    let future = previous_future
                        .join(acquire_future)
                        .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                        .unwrap()
                        .then_swapchain_present(
                            queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                        )
                        .then_signal_fence_and_flush();

                    fences[image_i as usize] = match future {
                        Ok(value) => Some(Arc::new(value)),
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            None
                        }
                        Err(e) => {
                            println!("failed to flush future: {e}");
                            None
                        }
                    };

                    previous_fence_i = image_i;
                }
                _ => {}
            }
        });
    }
}

fn main() {
    MainApplication::init();
}
