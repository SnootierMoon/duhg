use glam::{Mat4, Quat, Vec2, Vec3A};
use hexasphere::{
    shapes::{IcoSphere, IcoSphereBase},
    Subdivided,
};
use noise::{NoiseFn, Perlin};
use rand::Rng;
use std::time::Instant;
use wgpu::{util::DeviceExt, PresentMode};
use winit::{
    dpi::PhysicalPosition,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, MouseButton},
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::CursorGrabMode,
    window::{Window, WindowBuilder},
};

type Sphere = Subdivided<(), IcoSphereBase>;

struct State {
    window: Window,
    size: PhysicalSize<u32>,

    time: Instant,
    delta_time: f32,
    move_keys_held: [bool; 6],
    cursor_pos: Option<PhysicalPosition<f64>>,
    input_mode: InputMode,

    sphere: Sphere,
    camera: Camera,

    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth: Texture,
    num_indices: u32,
}

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        let camera = Camera {
            position: Vec3A::new(0.0, 2.0, 0.0),
            forward: Vec3A::new(1.0, 0.0, 0.0),
            pitch: 0.0,

            vel: Vec3A::ZERO,
            speed: 0.5,
            sens: 0.01,
            drag: 0.999,

            aspect_ratio: size.width as f32 / size.height as f32,
            fov_y_radians: std::f32::consts::FRAC_PI_4,
            z_near: 0.05,
            z_far: 20.0,
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let sphere = IcoSphere::new(1, |_| ());

        let min_radius = 0.7f32;
        let max_radius = 1.0f32;

        let n = Perlin::new(rand::thread_rng().gen());

        let vertices = sphere
            .raw_points()
            .iter()
            .map(|point| {
                let height = (n.get((point.as_dvec3() * 3.0).to_array()) / 2.0 + 0.5) as f32;
                let radius = (max_radius - min_radius) * height + min_radius;
                let point = *point * radius;
                Vertex {
                    position: [point.x, point.y, point.z],
                    color: [height, height, height],
                }
            })
            .collect::<Vec<_>>();

        let indices = sphere
            .get_all_indices()
            .into_iter()
            .map(|x| x as u16)
            .collect::<Vec<_>>();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&camera.matrix().to_cols_array()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let depth = Texture::create_depth_texture(&device, &config, "Depth");

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::DESC],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },

            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            window,
            size,
            time: Instant::now(),
            delta_time: 0.0,
            move_keys_held: [false; 6],
            cursor_pos: None,
            input_mode: InputMode::Cursor,
            sphere,
            camera,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            camera_bind_group,
            depth,
            num_indices: indices.len() as u32,
        }
    }

    pub fn set_input_mode(&mut self, input_mode: InputMode) {
        self.input_mode = input_mode;
        if input_mode == InputMode::Camera {
            _ = self
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked));
            self.window.set_cursor_visible(false);
        } else {
            _ = self.window.set_cursor_grab(CursorGrabMode::None);
            _ = self.window.set_cursor_position(PhysicalPosition::new(
                self.size.width / 2,
                self.size.height / 2,
            ));
            self.window.set_cursor_visible(true);
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.camera.aspect_ratio = new_size.width as f32 / new_size.height as f32;
            self.surface.configure(&self.device, &self.config);
            self.depth = Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn move_key(&self, key: VirtualKeyCode) -> Option<usize> {
        match key {
            VirtualKeyCode::W => Some(0),
            VirtualKeyCode::S => Some(1),
            VirtualKeyCode::A => Some(2),
            VirtualKeyCode::D => Some(3),
            VirtualKeyCode::Space => Some(4),
            VirtualKeyCode::LShift => Some(5),
            _ => None,
        }
    }

    fn update(&mut self) {
        let new_time = Instant::now();
        self.delta_time = new_time.duration_since(self.time).as_secs_f32();
        self.time = new_time;
        let mut move_vec = Vec3A::ZERO;
        if self.move_keys_held[0] {
            move_vec.x += 1.0;
        }
        if self.move_keys_held[1] {
            move_vec.x -= 1.0;
        }
        if self.move_keys_held[2] {
            move_vec.y += 1.0;
        }
        if self.move_keys_held[3] {
            move_vec.y -= 1.0;
        }
        if self.move_keys_held[4] {
            move_vec.z += 1.0;
        }
        if self.move_keys_held[5] {
            move_vec.z -= 1.0;
        }
        self.camera.update_position(move_vec, self.delta_time);

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera.matrix().to_cols_array()]),
        )
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Surface Texture View"),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn run(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    if self.input_mode == InputMode::Camera {
                        self.set_input_mode(InputMode::Cursor);
                    } else {
                        *control_flow = ControlFlow::Exit;
                    }
                }
                WindowEvent::Resized(physical_size)
                | WindowEvent::ScaleFactorChanged {
                    new_inner_size: &mut physical_size,
                    ..
                } => {
                    self.resize(physical_size);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.cursor_pos = Some(position);
                }
                WindowEvent::CursorLeft { device_id } => {
                    self.cursor_pos = None;
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    if self.input_mode == InputMode::Cursor {
                        self.set_input_mode(InputMode::Camera);
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Right,
                    ..
                } => {
                    if self.input_mode == InputMode::Cursor {
                        if let Some(pos) = self.cursor_pos {
                            let scaled_coords = 1.0
                                - 2.0
                                    * Vec2::new(
                                        pos.x as f32 / self.size.width as f32,
                                        pos.y as f32 / self.size.height as f32,
                                    );
                            self.camera.hit(&self.sphere, scaled_coords);
                        }
                    }
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                } => {
                    if let Some(id) = self.move_key(key) {
                        self.move_keys_held[id] = state == ElementState::Pressed;
                    }
                }
                _ => {}
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta, .. } => {
                    if self.input_mode == InputMode::Camera {
                        self.camera
                            .update_direction(Vec2::new(-delta.0 as f32, -delta.1 as f32));
                    }
                }
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == self.window.id() => {
                self.update();
                match self.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        self.resize(self.size)
                    }
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                }
            }
            Event::RedrawEventsCleared => {
                self.window.request_redraw();
            }
            _ => {}
        })
    }
}

struct Texture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float; // 1.

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&(label.to_string() + " Texture")),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&(label.to_string() + " View")),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&(label.to_string() + " Sampler")),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}

struct Camera {
    position: Vec3A,
    forward: Vec3A,
    pitch: f32, // [-pi/2, pi/2]

    vel: Vec3A,
    speed: f32,
    sens: f32,
    drag: f32,

    aspect_ratio: f32,
    fov_y_radians: f32,
    z_near: f32,
    z_far: f32,
}

impl Camera {
    fn matrix(&self) -> Mat4 {
        let up = self.position.normalize();
        let center = self.position + self.direction();
        let view = Mat4::look_at_rh(self.position.into(), center.into(), up.into());
        let proj = Mat4::perspective_rh(
            self.fov_y_radians,
            self.aspect_ratio,
            self.z_near,
            self.z_far,
        );
        proj * view
    }

    fn direction(&self) -> Vec3A {
        let (sin, cos) = self.pitch.sin_cos();
        self.forward * cos + self.position.normalize() * sin
    }

    fn left(&self) -> Vec3A {
        self.position.cross(self.forward).normalize()
    }

    // forward, left, up
    fn update_position(&mut self, accel: Vec3A, delta_time: f32) {
        self.vel = (1.0 - self.drag).powf(delta_time) * self.vel + self.speed * accel * delta_time;
        let h_move = self.vel.x * self.forward + self.vel.y * self.left();
        let rot_axis = self.position.cross(h_move).normalize_or_zero();
        let rot = Quat::from_axis_angle(
            rot_axis.into(),
            h_move.length() * self.position.length_recip(),
        );
        self.position = rot * self.position;
        self.forward = (rot * self.forward).reject_from(self.position).normalize();
        self.position += self.position.normalize() * self.vel.z;
    }

    // left, up
    fn update_direction(&mut self, delta: Vec2) {
        let delta_sens = delta * self.sens;
        self.pitch = (self.pitch + delta_sens.y).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.001,
            std::f32::consts::FRAC_PI_2 - 0.001,
        );
        let rot = Quat::from_axis_angle(self.position.normalize().into(), delta_sens.x);
        self.forward = (rot * self.forward).normalize();
    }

    fn hit(&mut self, sphere: &Sphere, cursor_coords: Vec2) -> Option<f32> {
        let tan_half_fov_y = (self.fov_y_radians * 0.5).tan();
        let dir = self.direction().normalize();
        let left = self.left().normalize();
        let up = dir.cross(left).normalize();
        let ray_dir = dir
            + (cursor_coords.x * left * self.aspect_ratio + cursor_coords.y * up) * tan_half_fov_y;
        let points = sphere.raw_points();
        sphere
            .get_all_indices()
            .chunks(3)
            .enumerate()
            .filter_map(|(i, tri)| {
                ray_tri_hit(
                    self.position,
                    ray_dir,
                    points[tri[0] as usize],
                    points[tri[1] as usize],
                    points[tri[2] as usize],
                )
                .map(|t| (t, i, tri[0], tri[1], tri[2]))
            })
            .min_by(|(t1, ..), (t2, ..)| t1.partial_cmp(t2).unwrap())
            .map(|(t, i, i0, i1, i2)| {
                println!("{i} {t}");
                t
            })
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    const DESC: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    };
}

fn ray_tri_hit(orig: Vec3A, dir: Vec3A, v0: Vec3A, v1: Vec3A, v2: Vec3A) -> Option<f32> {
    let (e1, e2) = (v1 - v0, v2 - v0);
    let h = dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-5 {
        return None;
    }
    let f = a.recip();
    let s = orig - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    if t < 0.0 {
        return None;
    }
    Some(t)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum InputMode {
    Camera,
    Cursor,
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Blohb")
        .build(&event_loop)
        .unwrap();

    pollster::block_on(async { State::new(window).await.run(event_loop) });
}
